"""Telegram bot message and callback handlers."""

from __future__ import annotations

import logging
import re

from telegram import Update
from telegram.ext import ContextTypes

from app import db as database
from app.answer_service import BotReply, build_reply
from app.bot.keyboards import clarify_choice_keyboard, feedback_keyboard
from app.config import (
    ADMIN_USER_IDS,
    ALLOWED_CHAT_IDS,
    ESCALATE_ON_NOT_HELPFUL,
    LOG_ALL_MESSAGES,
    OPERATOR_CHAT_ID,
    RAG_SHOW_FOOTER,
)
from app.retrieval import ResponseMode
from app.text_quality import button_label_from_question

logger = logging.getLogger(__name__)

_PENDING_ESC = re.compile(r"^#(\d+)\s+(.+)$", re.DOTALL)


def _reply_target(update: Update):
    """Message to reply to (works for both text messages and callback queries)."""
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return update.message


def _chat_allowed(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS.strip():
        return True
    allowed = {int(x.strip()) for x in ALLOWED_CHAT_IDS.split(",") if x.strip()}
    return chat_id in allowed


def _is_admin(user_id: int) -> bool:
    if not ADMIN_USER_IDS.strip():
        return False
    return str(user_id) in {x.strip() for x in ADMIN_USER_IDS.split(",") if x.strip()}


def _is_operator_chat(chat_id: int) -> bool:
    if not OPERATOR_CHAT_ID:
        return False
    try:
        return chat_id == int(OPERATOR_CHAT_ID)
    except ValueError:
        return False


def _log_in(update: Update, text: str) -> None:
    if not LOG_ALL_MESSAGES:
        return
    u = update.effective_user
    c = update.effective_chat
    if u and c:
        database.log_message(c.id, u.id, u.username, "in", text)


def _log_out(
    update: Update,
    text: str,
    reply: BotReply | None = None,
    faq_id: int | None = None,
) -> None:
    if not LOG_ALL_MESSAGES:
        return
    u = update.effective_user
    c = update.effective_chat
    if u and c:
        database.log_message(
            c.id,
            u.id,
            u.username,
            "out",
            text,
            mode=reply.mode.value if reply else None,
            faq_id=faq_id or (reply.faq_id if reply else None),
            similarity=reply.similarity if reply else None,
            used_rag=reply.used_rag if reply else False,
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    text = (
        "Здравствуйте! Я помощник техподдержки SmartTherm.\n\n"
        "Опишите проблему своими словами — я поищу ответ в базе знаний.\n"
        "Команды: /cancel — сбросить уточнение"
    )
    if _is_admin(update.effective_user.id):
        text += "\n/admin — статистика (для администратора)"
    await update.message.reply_text(text)


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        database.clear_context(update.effective_chat.id)
        database.clear_dialog_history(update.effective_chat.id)
    await update.message.reply_text("Контекст сброшен. Можете задать новый вопрос.")


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not _is_admin(update.effective_user.id):
        await update.message.reply_text("Недостаточно прав.")
        return
    stats = database.dashboard_stats()
    fb = stats["feedback"]
    lines = [
        "📊 Статистика бота",
        f"FAQ активных: {stats['faq_active']}",
        f"Запросов: {stats['queries_total']}",
        f"Эскалаций открытых: {stats['escalations_open']}",
        f"Feedback: 👍 {fb['helpful']} / 👎 {fb['not_helpful']}",
        "Режимы:",
    ]
    for row in stats.get("queries_by_mode", []):
        lines.append(f"  • {row['mode']}: {row['cnt']}")
    await update.message.reply_text("\n".join(lines))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_user or not update.message:
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    if _is_operator_chat(chat_id) and await _handle_operator_message(update, context, text):
        return

    if not _chat_allowed(chat_id):
        return

    _log_in(update, text)

    ctx = database.get_context(chat_id)
    if ctx and ctx.get("state") == "awaiting_clarification":
        combined = f"{ctx.get('pending_query', '')} {text}".strip()
        database.clear_context(chat_id)
        await _process_query(update, context, combined)
        return

    if ctx and ctx.get("state") == "awaiting_escalation_details":
        await _handle_escalation_followup(update, context, text, ctx)
        return

    await _process_query(update, context, text)


async def _handle_operator_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> bool:
    """
    Operator replies in operator chat:
      #123 <answer text>  — answer escalation #123 and optionally save draft
      /draft <question> ||| <answer>  — create FAQ draft
    """
    m = _PENDING_ESC.match(text.strip())
    if m:
        esc_id = int(m.group(1))
        answer = m.group(2).strip()
        esc = database.get_escalation(esc_id)
        if not esc:
            await update.message.reply_text(f"Эскалация #{esc_id} не найдена.")
            return True
        database.link_escalation_reply(esc_id, update.effective_user.id, answer)
        try:
            await context.bot.send_message(
                esc["chat_id"],
                f"Ответ эксперта по вашему вопросу:\n\n{answer}",
            )
        except Exception as e:
            logger.warning("Forward to user failed: %s", e)
            await update.message.reply_text(f"Не удалось доставить пользователю: {e}")
            return True
        database.create_faq_draft(
            question=esc["query_text"][:500],
            answer=answer,
            tags="from_escalation",
            source_escalation_id=esc_id,
        )
        await update.message.reply_text(
            f"✅ Ответ отправлен пользователю. Черновик FAQ создан (эскалация #{esc_id})."
        )
        return True

    if text.startswith("/draft "):
        body = text[7:].strip()
        if "|||" not in body:
            await update.message.reply_text("Формат: /draft вопрос ||| ответ")
            return True
        q, a = body.split("|||", 1)
        did = database.create_faq_draft(q.strip(), a.strip(), tags="operator")
        await update.message.reply_text(f"Черновик FAQ #{did} создан. Опубликуйте через админку.")
        return True

    return False


async def _handle_escalation_followup(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    ctx: dict,
) -> None:
    """User adds details after escalation — same ticket, retry search once."""
    chat_id = update.effective_chat.id
    esc_id = ctx.get("last_faq_id")
    pending = (ctx.get("pending_query") or "").strip()
    combined = f"{pending}\n{text}".strip() if pending else text

    if esc_id:
        database.append_escalation_details(int(esc_id), text)

    history = database.get_dialog_history(chat_id, limit=3)
    reply = build_reply(combined, history=history)
    if reply.mode == ResponseMode.AUTO:
        database.clear_context(chat_id)
        if esc_id:
            await update.message.reply_text(
                f"По уточнению нашёл ответ в базе знаний (обращение #{esc_id} можно закрыть):"
            )
        await _send_reply(update, context, combined, reply)
        return

    database.set_context(
        chat_id,
        update.effective_user.id,
        "awaiting_escalation_details",
        pending_query=combined,
        last_faq_id=esc_id,
    )
    note = (
        f"Спасибо, дополнение добавлено к обращению #{esc_id}.\n"
        "Эксперт ответит в чате поддержки."
        if esc_id
        else "Спасибо, передала эксперту. Ожидайте ответ."
    )
    await update.message.reply_text(note)
    if OPERATOR_CHAT_ID and esc_id:
        try:
            await context.bot.send_message(
                int(OPERATOR_CHAT_ID),
                f"📝 Уточнение к эскалации #{esc_id}:\n{text[:1500]}",
            )
        except Exception as e:
            logger.warning("Operator notify failed: %s", e)


async def _process_query(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    *,
    forced_faq_id: int | None = None,
) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if forced_faq_id is not None:
        card = database.get_faq(forced_faq_id)
        if not card:
            await update.message.reply_text("Карточка не найдена.")
            return
        reply = BotReply(
            mode=ResponseMode.AUTO,
            text=card["answer"],
            faq_id=card["id"],
            similarity=1.0,
            matches=[],
        )
        await _send_reply(update, context, query, reply)
        database.clear_context(chat_id)
        return

    history = database.get_dialog_history(chat_id, limit=3)
    reply = build_reply(query, history=history)
    log_text = reply.effective_query or query
    database.log_query(
        chat_id,
        user_id,
        log_text,
        reply.mode.value,
        reply.faq_id,
        reply.similarity,
    )

    if reply.mode == ResponseMode.CLARIFY:
        ids = [m.faq_id for m in reply.matches[:3]]
        labels = []
        for m in reply.matches[:3]:
            lbl = button_label_from_question(m.question)
            if lbl not in labels:
                labels.append(lbl)
        database.set_context(
            chat_id,
            user_id,
            "awaiting_clarification",
            pending_query=query,
            candidate_faq_ids=ids,
        )
        await update.message.reply_text(
            reply.text,
            reply_markup=clarify_choice_keyboard(ids, labels),
        )
        _log_out(update, reply.text, reply)
        return

    if reply.mode == ResponseMode.ESCALATE:
        await _escalate(update, context, query, reply)
        return

    await _send_reply(update, context, query, reply)
    database.clear_context(chat_id)


async def _send_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    reply: BotReply,
) -> None:
    msg = _reply_target(update)
    if not msg or not update.effective_user or not update.effective_chat:
        logger.error("Cannot send reply: no message target in update")
        return

    database.set_context(
        update.effective_chat.id,
        update.effective_user.id,
        "answered",
        last_faq_id=reply.faq_id,
    )
    context.user_data["last_query"] = query
    context.user_data["last_similarity"] = reply.similarity
    context.user_data["last_faq_id"] = reply.faq_id
    suffix = (
        "\n\n(ответ сформирован по базе знаний с помощью Ollama)"
        if reply.used_rag and RAG_SHOW_FOOTER
        else ""
    )
    text = reply.text + suffix

    if len(text) > 4000:
        text = text[:3997] + "..."
    await msg.reply_text(
        text,
        reply_markup=feedback_keyboard(reply.faq_id) if reply.faq_id else None,
    )
    _log_out(update, text, reply, reply.faq_id)
    try:
        database.push_dialog_turn(
            update.effective_chat.id,
            update.effective_user.id,
            query,
            reply.text,
            tier=reply.tier.value,
            faq_id=reply.faq_id,
        )
    except Exception:
        logger.exception("Failed to store dialog turn")


async def _escalate(
    update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, reply: BotReply
) -> None:
    chat_id = update.effective_chat.id
    user = update.effective_user
    candidates = [
        {
            "faq_id": m.faq_id,
            "question": m.question,
            "similarity": round(m.similarity, 3),
        }
        for m in reply.matches[:3]
    ]
    esc_id = database.create_escalation(
        user.id, chat_id, user.username, query, candidates
    )
    database.set_context(
        chat_id,
        user.id,
        "awaiting_escalation_details",
        pending_query=query,
        last_faq_id=esc_id,
    )
    msg = _reply_target(update)
    if msg:
        await msg.reply_text(reply.text)
    _log_out(update, reply.text, reply)

    if OPERATOR_CHAT_ID:
        try:
            op_chat = int(OPERATOR_CHAT_ID)
            cand_lines = "\n".join(
                f"• {c['similarity']:.2f} — {c['question'][:100]}" for c in candidates
            ) or "—"
            await context.bot.send_message(
                op_chat,
                f"🆘 Эскалация #{esc_id}\n"
                f"Ответьте: #{esc_id} <текст ответа>\n"
                f"От: @{user.username or user.id} (chat {chat_id})\n"
                f"Вопрос: {query}\n\n"
                f"Ближайшие FAQ:\n{cand_lines}",
            )
        except Exception as e:
            logger.warning("Failed to notify operator: %s", e)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.message:
        return

    data = query.data
    chat_id = query.message.chat_id
    user_id = query.from_user.id

    if not _chat_allowed(chat_id):
        await query.answer("Бот не обслуживает этот чат.", show_alert=True)
        return

    try:
        await _handle_callback(update, context, query, data, chat_id, user_id)
    except Exception:
        logger.exception("Callback handler failed: %s", data)
        await query.answer("Ошибка обработки. Попробуйте ещё раз.", show_alert=True)


async def _handle_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query,
    data: str,
    chat_id: int,
    user_id: int,
) -> None:
    await query.answer()

    if data.startswith("fb:"):
        _, helpful_s, faq_s = data.split(":", 2)
        helpful = helpful_s == "1"
        faq_id = int(faq_s)
        last_q = context.user_data.get("last_query", "")
        sim = context.user_data.get("last_similarity")
        database.save_feedback(faq_id, user_id, chat_id, last_q, helpful, sim)
        await query.edit_message_reply_markup(reply_markup=None)
        if helpful:
            await query.message.reply_text("Спасибо за отзыв!")
        else:
            await query.message.reply_text(
                "Жаль, что не помогло. Передаю эксперту — уточните детали одним сообщением."
            )
            if ESCALATE_ON_NOT_HELPFUL:
                fake_reply = BotReply(
                    mode=ResponseMode.ESCALATE,
                    text=(
                        "Передаю вопрос эксперту. Укажите модель котла и версию прошивки."
                    ),
                    faq_id=faq_id,
                    similarity=sim,
                    matches=[],
                )
                await _escalate(
                    update,
                    context,
                    last_q or "Не помогло (feedback)",
                    fake_reply,
                )
        return

    if data == "escalate":
        await query.edit_message_reply_markup(reply_markup=None)
        pending = ""
        ctx = database.get_context(chat_id)
        if ctx:
            pending = ctx.get("pending_query") or ""
        database.clear_context(chat_id)
        esc_text = pending or "Нужна помощь эксперта"
        esc_id = database.create_escalation(
            user_id, chat_id, query.from_user.username, esc_text, []
        )
        await query.message.reply_text(
            "Передаю вопрос эксперту. Укажите модель котла и версию прошивки."
        )
        if OPERATOR_CHAT_ID:
            try:
                await context.bot.send_message(
                    int(OPERATOR_CHAT_ID),
                    f"🆘 Эскалация #{esc_id}\nОтветьте: #{esc_id} <текст>\n"
                    f"От: @{query.from_user.username or user_id}\nВопрос: {esc_text}",
                )
            except Exception as e:
                logger.warning("Operator notify failed: %s", e)
        return

    if data.startswith("pick:"):
        faq_id = int(data.split(":")[1])
        ctx = database.get_context(chat_id)
        original = (ctx or {}).get("pending_query", "") or ""
        card = database.get_faq(faq_id)
        await query.edit_message_reply_markup(reply_markup=None)
        if not card:
            await query.message.reply_text(
                "Карточка не найдена. Опишите проблему текстом или нажмите /cancel."
            )
            return
        database.clear_context(chat_id)
        reply = BotReply(
            mode=ResponseMode.AUTO,
            text=card["answer"],
            faq_id=faq_id,
            similarity=None,
            matches=[],
        )
        await _send_reply(update, context, original, reply)
        return

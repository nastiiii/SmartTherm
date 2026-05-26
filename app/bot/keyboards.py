from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def feedback_keyboard(faq_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "👍 Помогло", callback_data=f"fb:1:{faq_id}"
                ),
                InlineKeyboardButton(
                    "👎 Не помогло", callback_data=f"fb:0:{faq_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "Позвать эксперта", callback_data="escalate"
                ),
            ],
        ]
    )


def clarify_choice_keyboard(faq_ids: list[int], labels: list[str]) -> InlineKeyboardMarkup:
    rows = []
    for fid, label in zip(faq_ids, labels):

        short = (label or f"Тема {fid}").strip()[:64]
        rows.append(
            [InlineKeyboardButton(short, callback_data=f"pick:{fid}")]
        )
    rows.append([InlineKeyboardButton("Ни один / другое", callback_data="escalate")])
    return InlineKeyboardMarkup(rows)

"""Hand-labeled agent requests; positives preserve the required output.

Negatives deliberately change one constraint or fact. Labels evaluate candidate
retrieval, not the correctness of model-generated answers. No LLM generates labels.
"""

FAMILIES = [
    ("python_dedup", "coding",
     "Write a Python function that removes duplicate strings from a list while preserving "
     "their original order. Treat uppercase and lowercase strings as different values.",
     ["Give me Python code to deduplicate a list of strings, keeping the first occurrence "
      "in its original position. Matching must be case-sensitive.",
      "I need a Python helper that keeps each string only once without changing input order; "
      "strings with different capitalization should remain distinct."],
     ["Write a Python function that removes duplicate strings from a list while preserving "
      "their original order. Treat uppercase and lowercase strings as the same value.",
      "Write a Python function that removes duplicate strings from a list and sorts the "
      "result alphabetically. Treat uppercase and lowercase strings as different values."]),
    ("sql_latest", "coding",
     "Write PostgreSQL SQL to get the latest order for each customer from "
     "orders(customer_id, order_id, created_at). Break timestamp ties using the largest order_id.",
     ["Using PostgreSQL and orders(customer_id, order_id, created_at), return each customer's "
      "most recent order. If timestamps tie, choose the greatest order_id.",
      "Give me a Postgres query selecting the newest row per customer in "
      "orders(customer_id, order_id, created_at), with highest order_id winning equal timestamps."],
     ["Write PostgreSQL SQL to get the earliest order for each customer from "
      "orders(customer_id, order_id, created_at). Break timestamp ties using the largest order_id.",
      "Write PostgreSQL SQL to get the latest order for each customer from "
      "orders(customer_id, order_id, created_at). Return all rows when timestamps tie."]),
    ("python_retry", "coding",
     "Write an async Python helper that retries an HTTP GET when a connection times out. "
     "Use exponential backoff, allow at most three total attempts, and propagate the final error.",
     ["Show an asyncio HTTP GET retry helper for connection timeouts with exponential "
      "backoff. Make three attempts maximum and raise the last failure if none succeeds.",
      "Implement a coroutine that retries a timed-out GET with exponentially increasing "
      "waits, stops after three total tries, and re-raises the final error."],
     ["Write an async Python helper that retries an HTTP POST when a connection times out. "
      "Use exponential backoff, allow at most three total attempts, and propagate the final error.",
      "Write an async Python helper that retries an HTTP GET when a connection times out. "
      "Use exponential backoff, allow at most three total attempts, and return None on failure."]),
    ("js_dates", "coding",
     "Write JavaScript that sorts objects by their ISO date field newest first. "
     "Keep objects with missing dates at the end and do not mutate the original array.",
     ["Provide JavaScript to order records descending by ISO date, putting undated records "
      "last while leaving the input array unchanged.",
      "I want JS code that returns a new array sorted by date from most recent to oldest, "
      "with missing date values placed after dated items."],
     ["Write JavaScript that sorts objects by their ISO date field oldest first. "
      "Keep objects with missing dates at the end and do not mutate the original array.",
      "Write JavaScript that sorts objects by their ISO date field newest first. "
      "Keep objects with missing dates at the beginning and do not mutate the original array."]),
    ("regex_filename", "coding",
     "Give a Python regex that matches an entire filename ending in .csv, ignoring case. "
     "Reject names containing directory separators. Explain the expression briefly.",
     ["Show and briefly explain a case-insensitive Python regular expression for a bare "
      "CSV filename. It must match the whole name and disallow slash or backslash.",
      "I need a full-filename Python pattern accepting the .csv extension in any case, "
      "excluding paths. Include a short explanation."],
     ["Give a Python regex that matches an entire filename ending in .csv, respecting case. "
      "Reject names containing directory separators. Explain the expression briefly.",
      "Give a Python regex that matches an entire filename ending in .json, ignoring case. "
      "Reject names containing directory separators. Explain the expression briefly."]),
    ("bug_mutable", "coding",
     "Explain why this Python code shares state between calls and show the fix: "
     "def add_item(item, items=[]): items.append(item); return items. Keep the public signature "
     "compatible for callers that supply a list.",
     ["Diagnose and fix the shared-list bug in Python: def add_item(item, items=[]): "
      "items.append(item); return items. Existing callers passing a list must still work.",
      "Why does def add_item(item, items=[]): items.append(item); return items retain earlier "
      "values? Explain and repair it without breaking callers that explicitly pass items."],
     ["Explain why this Python code shares state between calls and show the fix: "
      "def add_item(item, items=[]): items.append(item); return items. Convert the API "
      "to a class with an instance-owned list.",
      "Explain why this Python code shares state between calls: def add_item(item, items=[]): "
      "items.append(item); return items. Give only an explanation, without code."]),
    ("refund_email", "support",
     "Draft a friendly support email to Maya. Her unused subscription renewal is refundable "
     "under our policy, and the refund has been approved. Tell her it will return to her "
     "original payment method. Do not promise a processing date.",
     ["Write Maya a warm customer-service message confirming approval of a refund for her "
      "unused subscription renewal. Say it goes back to the original payment method; "
      "include no timeline commitment.",
      "Compose a friendly email for Maya about her approved unused-renewal refund. "
      "Confirm repayment to the method she paid with, without estimating when it arrives."],
     ["Draft a friendly support email to Noah. His unused subscription renewal is refundable "
      "under our policy, and the refund has been approved. Tell him it will return to his "
      "original payment method. Do not promise a processing date.",
      "Draft a friendly support email to Maya. Her used subscription renewal is not refundable "
      "under our policy, and the refund has been denied. Explain the decision politely."]),
    ("policy_refund", "support",
     "According to this policy, can an opened but defective item be refunded? Policy: "
     "Unopened items qualify for returns. Defective items qualify even if opened. "
     "Final-sale items are excluded unless defective. Answer briefly.",
     ["Give a short answer: is an opened defective product eligible for a refund under "
      "these rules? Unopened goods can be returned; defective goods can be returned opened; "
      "final-sale goods are excluded except when defective.",
      "Can I refund a faulty item after opening it? Briefly apply this policy: unopened "
      "items are returnable, defects permit opened-item returns, and final-sale exceptions "
      "apply only to defects."],
     ["According to this policy, can an opened non-defective item be refunded? Policy: "
      "Unopened items qualify for returns. Defective items qualify even if opened. "
      "Final-sale items are excluded unless defective. Answer briefly.",
      "According to this policy, can an opened but defective item be refunded? Policy: "
      "Only unopened items qualify for refunds, including defective items. Answer briefly."]),
    ("translate", "writing",
     "Translate this message into Spanish in a formal business tone: 'We received your "
     "application and will contact you after reviewing the documents.' "
     "Return only the translation.",
     ["Provide only a formal Spanish translation of: 'We received your application and "
      "will contact you after reviewing the documents.'",
      "Render the following in professional, formal Spanish, with no commentary: 'We "
      "received your application and will contact you after reviewing the documents.'"],
     ["Translate this message into French in a formal business tone: 'We received your "
      "application and will contact you after reviewing the documents.' "
      "Return only the translation.",
      "Translate this message into Spanish in a formal business tone: 'We rejected your "
      "application after reviewing the documents.' Return only the translation."]),
    ("meeting", "writing",
     "Summarize these meeting notes into action items with owner names: Alex will update "
     "the onboarding guide; Priya will investigate checkout failures; Sam will schedule "
     "the release review. Return a bullet list.",
     ["Turn these meeting notes into bullets listing tasks and responsible people: Alex "
      "will update the onboarding guide; Priya will investigate checkout failures; Sam "
      "will schedule the release review.",
      "Extract owner-tagged follow-up actions as a bullet list from these notes: Alex "
      "will update the onboarding guide; Priya will investigate checkout failures; Sam "
      "will schedule the release review."],
     ["Summarize these meeting notes into action items with owner names: Priya will update "
      "the onboarding guide; Alex will investigate checkout failures; Sam will schedule "
      "the release review. Return a bullet list.",
      "Summarize these meeting notes into action items with owner names: Alex will update "
      "the onboarding guide; Priya will investigate checkout failures; Sam will schedule "
      "the release review. Return a JSON object, not a bullet list."]),
    ("incident", "operations",
     "Write an internal incident summary from these facts: checkout was unavailable; "
     "a misconfigured feature flag caused the outage; reverting the flag restored service. "
     "Include impact, cause, and mitigation. Do not invent a timeline.",
     ["Create an internal outage report covering impact, root cause and resolution: "
      "checkout was unavailable; a misconfigured feature flag caused the outage; reverting "
      "the flag restored service. Add no unsupported timing details.",
      "Summarize this incident internally, identifying impact, cause and fix without "
      "making up times: checkout was unavailable; a misconfigured feature flag caused "
      "the outage; reverting the flag restored service."],
     ["Write an internal incident summary from these facts: checkout was unavailable; "
      "a database connection leak caused the outage; restarting the service restored "
      "service. Include impact, cause, and mitigation. Do not invent a timeline.",
      "Write a customer-facing apology from these facts: checkout was unavailable; "
      "a misconfigured feature flag caused the outage; reverting the flag restored service. "
      "Avoid internal technical details."],
    ),
    ("docker", "operations",
     "Give the Docker command to list running containers, showing their names and images "
     "only. Do not include stopped containers.",
     ["How do I show just container names and image names for currently running Docker "
      "containers? Provide the command and exclude stopped ones.",
      "Return a Docker CLI command displaying only names and images of active containers; "
      "omit exited containers."],
     ["Give the Docker command to list all containers, showing their names and images "
      "only. Include stopped containers.",
      "Give the Docker command to delete stopped containers. Do not affect running containers."]),
    ("git", "operations",
     "Show how to undo the most recent Git commit while keeping its changes staged. "
     "Assume the commit has not been pushed. Explain the command briefly.",
     ["I have an unpushed last commit. Give and briefly explain the Git command to remove "
      "that commit but leave its edits in the staging area.",
      "How can I roll back my latest local Git commit without unstaging its changes? "
      "It is not on the remote yet; include a short explanation."],
     ["Show how to undo the most recent Git commit while discarding all its changes. "
      "Assume the commit has not been pushed. Explain the command briefly.",
      "Show how to undo the most recent Git commit while keeping its changes unstaged. "
      "Assume the commit has not been pushed. Explain the command briefly."]),
    ("logs", "operations",
     "Explain the likely cause and next debugging step for this log: 'password "
     "authentication failed for user app_user' when a service connects to PostgreSQL. "
     "Do not suggest disabling authentication.",
     ["A service connecting to Postgres logs 'password authentication failed for user "
      "app_user'. What probably went wrong and what should I check next? "
      "Keep authentication enabled.",
      "Diagnose the PostgreSQL connection error 'password authentication failed for "
      "user app_user' and recommend the next check without bypassing login security."],
     ["Explain the likely cause and next debugging step for this log: 'connection refused' "
      "when a service connects to PostgreSQL. Do not suggest disabling authentication.",
      "Explain the likely cause and next debugging step for this log: 'permission denied "
      "for table orders' when a service queries PostgreSQL. "
      "Do not suggest disabling authentication."]),
    ("invoice", "data",
     "Extract invoice number, supplier, and currency as JSON from this text: "
     "'Invoice INV-ALPHA from Northwind Supplies. Currency: EUR.' Do not add extra keys.",
     ["Return only JSON containing invoice number, supplier and currency for: "
      "'Invoice INV-ALPHA from Northwind Supplies. Currency: EUR.'",
      "Parse 'Invoice INV-ALPHA from Northwind Supplies. Currency: EUR.' into a JSON "
      "object with only invoice number, supplier and currency."],
     ["Extract invoice number, supplier, and currency as JSON from this text: "
      "'Invoice INV-ALPHA from Northwind Supplies. Currency: USD.' Do not add extra keys.",
      "Extract invoice number, supplier, and currency as JSON from this text: "
      "'Invoice INV-BETA from Contoso Supplies. Currency: EUR.' Do not add extra keys."]),
    ("classification", "data",
     "Classify this customer message as billing, technical, or cancellation. Return only "
     "the category: 'I was charged twice for my subscription renewal.'",
     ["Choose billing, technical, or cancellation for this message, outputting just the "
      "label: 'I was charged twice for my subscription renewal.'",
      "Label the ticket 'I was charged twice for my subscription renewal.' using only "
      "one category from billing, technical, cancellation."],
     ["Classify this customer message as billing, technical, or cancellation. Return only "
      "the category: 'I want to stop my subscription before it renews.'",
      "Classify this customer message as billing, technical, or cancellation. Return only "
      "the category: 'I cannot log in to my account.'"]),
    ("csv", "data",
     "Explain how to read a UTF-encoded CSV file in pandas while preserving the order_id "
     "column as strings, including leading zeros. Give a short code example.",
     ["Show a brief pandas CSV loading example that treats order_id as text so leading "
      "zeros survive; the file uses UTF encoding.",
      "How can pandas import a UTF-encoded comma-separated file without converting "
      "order_id identifiers to numbers? Preserve leading zeros and include concise code."],
     ["Explain how to read a UTF-encoded CSV file in pandas while converting the order_id "
      "column to integers and removing leading zeros. Give a short code example.",
      "Explain how to read a UTF-encoded CSV file in pandas while preserving the account_id "
      "column as strings, including leading zeros. Give a short code example."]),
    ("percentage", "data",
     "Calculate the percentage increase when monthly revenue rises from 12000 to 15000. "
     "Show the formula and the result, without rounding intermediate values.",
     ["Monthly revenue went from 12000 to 15000. Work out the percentage growth, "
      "showing the formula and exact result without intermediate rounding.",
      "Find the percent change for a revenue increase from 12000 to 15000, giving "
      "both the calculation formula and the result without rounding during calculation."],
     ["Calculate the percentage increase when monthly revenue rises from 12000 to 18000. "
      "Show the formula and the result, without rounding intermediate values.",
      "Calculate the percentage decrease when monthly revenue falls from 15000 to 12000. "
      "Show the formula and the result, without rounding intermediate values."]),
    ("portfolio", "finance",
     "Using only this snapshot, calculate portfolio ALPHA's total USD exposure: "
     "USD assets 1000, EUR assets 2000, and EUR/USD 1.1. Show the calculation.",
     ["Calculate total USD exposure for portfolio ALPHA from this snapshot only: "
      "USD assets 1000, EUR assets 2000, and EUR/USD 1.1. Include the arithmetic.",
      "What is portfolio ALPHA worth in USD using these supplied values only: "
      "USD assets 1000, EUR assets 2000, and EUR/USD 1.1? Show your calculation."],
     ["Using only this snapshot, calculate portfolio BETA's total USD exposure: "
      "USD assets 1000, EUR assets 2000, and EUR/USD 1.1. Show the calculation.",
      "Using only this snapshot, calculate portfolio ALPHA's total USD exposure: "
      "USD assets 1000, EUR assets 2000, and EUR/USD 1.2. Show the calculation."]),
    ("access", "security",
     "Explain why a service account should receive read-only access to the reporting "
     "database instead of administrator access. Focus on least privilege and incident impact.",
     ["Why is read-only reporting-database access safer for a service account than admin "
      "rights? Explain using least privilege and potential incident damage.",
      "Describe the least-privilege rationale and reduced incident impact of granting "
      "a reporting service account read-only database permissions rather than "
      "administrator rights."],
     ["Give SQL commands to grant a service account administrator access to the reporting "
      "database instead of read-only access.",
      "Explain why a human database administrator needs write access to the production "
      "database. Focus on operational maintenance tasks."]),
]

NOVEL = [
    "Plan a vegetarian dinner menu for a family that avoids nuts, with a shopping list.",
    "Explain photosynthesis to a primary-school student using an everyday analogy.",
    "Help me structure a product launch announcement for a new bicycle helmet.",
    "Write a bedtime story about a fox who learns to share with woodland friends.",
    "Create an interview rubric for hiring a senior mobile application developer.",
    "Suggest a weekly language-learning routine for practicing spoken Japanese.",
    "Draft a checklist for moving from an apartment into a house with a pet cat.",
    "Explain the difference between weather and climate for a classroom presentation.",
]


def requests():
    rows = []
    for key, category, canonical, positives, negatives in FAMILIES:
        for index, text in enumerate(positives):
            rows.append({"id": f"{key}:positive:{index}", "family": key, "category": category,
                         "text": text, "expected_id": key, "label": "positive"})
        for index, text in enumerate(negatives):
            rows.append({"id": f"{key}:negative:{index}", "family": key, "category": category,
                         "text": text, "expected_id": None, "label": "hard_negative"})
    rows.extend({"id": f"novel:{i}", "family": None, "category": "novel", "text": text,
                 "expected_id": None, "label": "novel"} for i, text in enumerate(NOVEL))
    return rows

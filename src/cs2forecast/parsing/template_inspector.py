from collections import Counter

import mwparserfromhell


def count_templates(wikitext: str) -> Counter[str]:
    wikicode = mwparserfromhell.parse(wikitext)

    counter: Counter[str] = Counter()

    for template in wikicode.filter_templates(recursive=True):
        name = str(template.name).strip()
        counter[name] += 1

    return counter


def format_template_examples(wikitext: str, template_name: str, limit: int = 5) -> list[str]:
    wikicode = mwparserfromhell.parse(wikitext)
    examples: list[str] = []

    wanted = template_name.lower().strip()

    for template in wikicode.filter_templates(recursive=True):
        name = str(template.name).strip().lower()

        if name != wanted:
            continue

        lines = [f"Template: {template.name}"]

        for param in template.params[:30]:
            lines.append(f"  {str(param.name).strip()} = {str(param.value).strip()}")

        examples.append("\n".join(lines))

        if len(examples) >= limit:
            break

    return examples
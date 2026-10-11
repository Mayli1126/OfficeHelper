import re
from dataclasses import dataclass

import config_data as config


@dataclass(frozen=True)
class VerificationReport:
    answer: str
    passed: bool
    issues: tuple[str, ...]


class AnswerVerifierAgent:
    REQUIRED_SECTIONS = ("最终回答", "操作步骤/材料清单", "风险提示")

    def verify(self, answer, references):
        if not references:
            return VerificationReport(config.NO_EVIDENCE_MESSAGE, True, ("no_evidence",))

        cleaned = answer.strip()
        issues = []
        for section in self.REQUIRED_SECTIONS:
            if f"### {section}" not in cleaned:
                issues.append(f"missing_section:{section}")
                cleaned += f"\n\n### {section}\n无"

        citation_ids = {int(value) for value in re.findall(r"\[(\d+)\]", cleaned)}
        valid_ids = set(range(1, len(references) + 1))
        if not citation_ids:
            issues.append("missing_inline_citation")
        elif not citation_ids.issubset(valid_ids):
            issues.append("invalid_inline_citation")
            cleaned = re.sub(
                r"\[(\d+)\]",
                lambda match: match.group(0) if int(match.group(1)) in valid_ids else "",
                cleaned,
            )

        return VerificationReport(cleaned, not issues, tuple(issues))

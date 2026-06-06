"""
Code explanation, bug detection, and complexity analysis.
  - Auto-detects programming language
  - Explains what the code does
  - Detects bugs / issues
  - Reports time + space complexity
"""

import logging
import re

import google.generativeai as genai
from langchain.prompts import PromptTemplate

logger = logging.getLogger("omni-agent-ai.tools.code_explain")

LANGUAGE_PATTERNS = {
    "Python":     [r"def ", r"import ", r"print\(", r":\n", r"elif "],
    "JavaScript": [r"const ", r"let ", r"var ", r"=>", r"console\.log"],
    "TypeScript": [r"interface ", r": string", r": number", r"=> \{"],
    "Java":       [r"public class", r"System\.out", r"void main", r"@Override"],
    "C++":        [r"#include", r"std::", r"cout <<", r"int main\(\)"],
    "SQL":        [r"SELECT ", r"FROM ", r"WHERE ", r"JOIN ", r"INSERT INTO"],
    "Go":         [r"func ", r"package main", r":= ", r"fmt\."],
    "Rust":       [r"fn main", r"let mut", r"println!", r"impl "],
}

CODE_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["language", "code"],
    template="""You are a senior software engineer and code reviewer.

Analyze the following {language} code and provide:

**WHAT IT DOES:**
<Clear explanation of the code's purpose and logic, step by step>

**DETECTED BUGS / ISSUES:**
<List any bugs, edge cases, or bad practices. Write "None detected" if clean>

**TIME COMPLEXITY:** <O(?) with explanation>
**SPACE COMPLEXITY:** <O(?) with explanation>

**IMPROVED VERSION (optional):**
<Only include if there are meaningful improvements to suggest>

Code:
```{language}
{code}
```

Be specific, technical, and helpful."""
)


async def explain_code(text: str) -> str:
    """
    Explain code extracted from an image or text input.
    Auto-detects language, explains, finds bugs, reports complexity.
    """
    if not text or len(text.strip()) < 10:
        return "[No code content found to explain.]"

    logger.info(f"Explaining code: {len(text)} chars")

    #Extract code block if mixed
    code = _extract_code_block(text)
    detected = _detect_language(code)
    if detected == "Unknown":
        language = "the programming language used in this code"
    else:
        language = detected

    logger.info(f"Detected language:{language}")

    prompt=CODE_EXPLAIN_PROMPT.format(language=language,code=code[:4000])
    model=genai.GenerativeModel("gemini-2.5-flash")

    response=model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1000,
        ),
    )
    result=response.text.strip()
    logger.info(f"code explanation generated:{len(result)}chars")

    return f"🔍 **Detected Language:{language}**\n\n{result}"


def _detect_language(code: str) -> str:
    """
    Detect programming language using regex pattern matching.
    """
    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score=sum(1 for p in patterns if re.search(p, code))
        if score>0:
            scores[lang]=score

    if not scores:
        return "Unknown"

    return max(scores,key=scores.get)


def _extract_code_block(text: str) -> str:
    """
    If text contains markdown code fences, extract just the code.
    Otherwise return the full text (it's probably raw code from OCR).
    """
    fence_match= re.search(r"```[\w]*\n(.*?)```",text,re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    lines=text.split("\n")
    code_lines=[l for l in lines if l.startswith("    ") or l.startswith("\t")]
    if len(code_lines)>len(lines)*0.4:
        return "\n".join(l.lstrip() for l in code_lines)

    return text.strip()
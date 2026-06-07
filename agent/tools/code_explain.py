"""Code explanation and complexity analysis."""
import logging
import re
from agent.config import GEMINI_MODEL, get_client

logger=logging.getLogger("omni-agent-ai.tools.code_explain")

lang_patterns={
    "Python":[r"def ",r"import ",r"print\(",r":\n",r"elif "],
    "JavaScript":[r"const ",r"let ",r"var ",r"=>",r"console\.log"],
    "TypeScript":[r"interface ",r": string",r": number",r"=> \{"],
    "Java":[r"public class",r"System\.out",r"void main",r"@Override"],
    "C++":[r"#include",r"std::",r"cout <<",r"int main\(\)"],
    "SQL":[r"SELECT ",r"FROM ",r"WHERE ",r"JOIN ",r"INSERT INTO"],
    "Go":[r"func ",r"package main",r":= ",r"fmt\."],
    "Rust":[r"fn main",r"let mut",r"println!",r"impl "],
}

prompt_template="""Explain this {language} code.
Please structure your review as follows:
- What the code does: Step-by-step description of the logic.
- Bugs or issues: Any issues, edge cases, or potential fixes (or write "None detected" if clean).
- Complexity: Time and space complexity details.
- Suggested improvements (if any).

Here is the code:
```{language}
{code}
```"""

async def explain_code(txt:str) -> str:
    if not txt or len(txt.strip())<10:
        return "[No code content found to explain.]"

    logger.info(f"Explaining code: {len(txt)} chars")
    code=_get_code(txt)
    lang=_get_lang(code)
    label=lang if lang!="Unknown" else "the code"

    prompt=prompt_template.format(language=label,code=code[:4000])
    client=get_client()
    resp=client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    res=resp.text.strip()
    return f"🔍 **Detected Language: {lang}**\n\n{res}"

def _get_lang(code:str) -> str:
    scores={}
    for lang,patterns in lang_patterns.items():
        score=sum(1 for p in patterns if re.search(p,code))
        if score>0:
            scores[lang]=score
    if not scores:
        return "Unknown"
    return max(scores,key=scores.get)

def _get_code(txt:str) -> str:
    match=re.search(r"```[\w]*\n(.*?)```",txt,re.DOTALL)
    if match:
        return match.group(1).strip()

    lines=txt.split("\n")
    code_lines=[l for l in lines if l.startswith("    ") or l.startswith("\t")]
    if len(code_lines)>len(lines)*0.4:
        return "\n".join(l.lstrip() for l in code_lines)

    return txt.strip()
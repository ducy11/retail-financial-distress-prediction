"""Convert the generated Word report into LaTeX from the same `scripts.make_uit_report` blocks.

Reads `BLOCKS` so the LaTeX output matches the .docx content exactly, copies the referenced figures, and
writes `docs/latex_baocao/{main.tex, figures/*.png}`. Compile with XeLaTeX.
"""
from __future__ import annotations

import importlib
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = importlib.import_module("scripts.make_uit_report").BLOCKS

OUT = ROOT / "docs" / "latex_baocao"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# Copy the referenced figures into the LaTeX bundle.
for rel in sorted({b[1][0] for b in BLOCKS if b[0] == "img"}):
    src = ROOT / rel
    if src.exists():
        shutil.copy2(src, OUT / "figures" / src.name)

# Preamble, mirroring the formatting of the generated Word file.
PREAMBLE = r"""% =====================================================================
%  BÁO CÁO ĐỒ ÁN CUỐI KỲ — UIT (ĐHQG-HCM)
%  Bản LaTeX chuyển tự động từ Bao_Cao_Cuoi_Ky_UIT.docx
%  Biên dịch: XeLaTeX (xelatex -> bibtex -> xelatex -> xelatex)
%  Định dạng: A4, lề Trên/Dưới 2.0cm · Trái 3.0cm · Phải 2.0cm, TNR 13pt, dãn dòng 1.4
% =====================================================================
\documentclass[13pt,a4paper]{extreport}

\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage[vietnamese]{babel}
\usepackage{amsmath,amssymb}

\usepackage[a4paper,top=2.0cm,bottom=2.0cm,left=3.0cm,right=2.0cm]{geometry}

\usepackage{setspace}
\setstretch{1.4}
\setlength{\parindent}{1.0cm}
\setlength{\parskip}{6pt}

\usepackage{graphicx}
\usepackage{float}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage[font=small,labelfont=bf,justification=centering]{caption}

\usepackage{indentfirst}
\usepackage{enumitem}
\setlist{nosep,leftmargin=1.2cm}

\usepackage{titlesec}
\titleformat{\chapter}[block]
  {\normalfont\bfseries\centering\fontsize{15}{18}\selectfont}
  {CHƯƠNG \thechapter.}{0.5em}{}
\titlespacing*{\chapter}{0pt}{0pt}{16pt}
\titleformat{\section}{\normalfont\bfseries\fontsize{13}{16}\selectfont}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titleformat{\subsection}{\normalfont\bfseries\fontsize{13}{16}\selectfont}{\thesubsection.}{0.5em}{}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\cfoot{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\usepackage[hidelinks,unicode]{hyperref}

\renewcommand{\contentsname}{MỤC LỤC}
\renewcommand{\listfigurename}{DANH SÁCH BIỂU ĐỒ}
\renewcommand{\listtablename}{DANH SÁCH BẢNG}
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}

\begin{document}

% ---------------------------------------------------------------------
%  TRANG BÌA
% ---------------------------------------------------------------------
\begin{titlepage}
\centering
{\bfseries\fontsize{13}{16}\selectfont ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\par}
\vspace{2pt}
{\bfseries\fontsize{14}{17}\selectfont TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN\par}
\vspace{6pt}
{\bfseries\fontsize{13}{16}\selectfont KHOA HỆ THỐNG THÔNG TIN\par}
\vspace{2.4cm}
{\bfseries\fontsize{15}{18}\selectfont BÁO CÁO ĐỒ ÁN CUỐI KỲ\par}
\vspace{0.3cm}
{\fontsize{13}{16}\selectfont Môn học: <TÊN MÔN HỌC> (mã lớp CS114)\par}
\vspace{1.0cm}
{\bfseries\fontsize{13}{16}\selectfont ĐỀ TÀI\par}
\vspace{0.4cm}
{\bfseries\fontsize{15}{18}\selectfont DỰ BÁO SUY GIẢM TÀI CHÍNH DOANH NGHIỆP\\[2pt]
BÁN LẺ TỪ DỮ LIỆU SEC XBRL BẰNG HỌC MÁY\par}
\vspace{1.4cm}
\begin{flushleft}
{\fontsize{13}{16}\selectfont Giảng viên hướng dẫn: <ThS. …………………>\par}
\vspace{0.4cm}
{\fontsize{13}{16}\selectfont Sinh viên thực hiện:\par}
\vspace{0.2cm}
{\fontsize{13}{16}\selectfont <Họ và tên> --- <MSSV>\par}
{\fontsize{13}{16}\selectfont <Họ và tên> --- <MSSV>\par}
{\fontsize{13}{16}\selectfont <Họ và tên> --- <MSSV>\par}
{\fontsize{13}{16}\selectfont <Họ và tên> --- <MSSV>\par}
\end{flushleft}
\vfill
{\bfseries\fontsize{13}{16}\selectfont TP. Hồ Chí Minh, tháng <…> năm <…>\par}
\end{titlepage}
\clearpage
"""

# References, keeping the [1]..[17] numbering of the Word file.
REFS = r"""
\chapter*{TÀI LIỆU THAM KHẢO}
\begin{thebibliography}{99}
\bibitem{beaver1966} W. H. Beaver, ``Financial ratios as predictors of failure,'' \emph{Journal of
Accounting Research}, vol. 4, pp. 71--111, 1966.
\bibitem{altman1968} E. I. Altman, ``Financial ratios, discriminant analysis and the prediction of
corporate bankruptcy,'' \emph{The Journal of Finance}, vol. 23, no. 4, pp. 589--609, 1968.
\bibitem{altman2000} E. I. Altman, ``Predicting financial distress of companies: Revisiting the
Z-score and ZETA models,'' Working paper, 2000.
\bibitem{barboza2017} F. Barboza, H. Kimura, and E. Altman, ``Machine learning models and bankruptcy
prediction,'' \emph{Expert Systems with Applications}, vol. 83, pp. 405--417, 2017.
\bibitem{pedregosa2011} F. Pedregosa et al., ``Scikit-learn: Machine learning in Python,''
\emph{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011.
\bibitem{breiman2001} L. Breiman, ``Random forests,'' \emph{Machine Learning}, vol. 45, no. 1,
pp. 5--32, 2001.
\bibitem{friedman2001} J. H. Friedman, ``Greedy function approximation: A gradient boosting
machine,'' \emph{Annals of Statistics}, vol. 29, no. 5, pp. 1189--1232, 2001.
\bibitem{chen2016} T. Chen and C. Guestrin, ``XGBoost: A scalable tree boosting system,'' in
\emph{Proc. ACM SIGKDD}, 2016, pp. 785--794.
\bibitem{fawcett2006} T. Fawcett, ``An introduction to ROC analysis,'' \emph{Pattern Recognition
Letters}, vol. 27, no. 8, pp. 861--874, 2006.
\bibitem{davis2006} J. Davis and M. Goadrich, ``The relationship between Precision-Recall and ROC
curves,'' in \emph{Proc. ICML}, 2006, pp. 233--240.
\bibitem{delong1988} E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, ``Comparing the areas
under two or more correlated receiver operating characteristic curves,'' \emph{Biometrics}, vol. 44,
no. 3, pp. 837--845, 1988.
\bibitem{chawla2002} N. V. Chawla et al., ``SMOTE: Synthetic Minority Over-sampling Technique,''
\emph{Journal of Artificial Intelligence Research}, vol. 16, pp. 321--357, 2002.
\bibitem{lemaitre2017} G. Lema\^{i}tre, F. Nogueira, and C. K. Aridas, ``Imbalanced-learn: A Python
toolbox to tackle the curse of imbalanced datasets,'' \emph{Journal of Machine Learning Research},
vol. 18, pp. 1--5, 2017.
\bibitem{lundberg2017} S. M. Lundberg and S.-I. Lee, ``A unified approach to interpreting model
predictions,'' in \emph{Proc. NeurIPS}, 2017, pp. 4765--4774.
\bibitem{fisher2019} A. Fisher, C. Rudin, and F. Dominici, ``All models are wrong, but many are
useful: Learning a variable's importance by studying an entire class of prediction models,''
\emph{Journal of Machine Learning Research}, vol. 20, pp. 1--81, 2019.
\bibitem{secxbrl2026} U.S. Securities and Exchange Commission, ``Structured data / XBRL,''
\url{https://www.sec.gov/structureddata} (truy cập 2026).
\bibitem{secfacts2026} U.S. Securities and Exchange Commission, ``SEC EDGAR XBRL company facts
API,'' \url{https://data.sec.gov/api/xbrl/companyfacts/} (truy cập 2026).
\end{thebibliography}
"""

# Text-to-LaTeX helpers.
_TEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _plain(s: str) -> str:
    return "".join(_TEX_MAP.get(c, c) for c in s)


def tex(s: str) -> str:
    """Map bold markers to \\textbf, `code` -> \\texttt, and escape LaTeX special characters."""
    out = []
    for token in re.split(r"(\*\*.+?\*\*|`[^`]+`)", s):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            out.append(r"\textbf{" + _plain(token[2:-2]) + "}")
        elif token.startswith("`") and token.endswith("`"):
            out.append(r"\texttt{" + _plain(token[1:-1]) + "}")
        else:
            out.append(_plain(token))
    return "".join(out)


def strip_num(text: str, level: int) -> str:
    """Strip the section number prefix such as '1.2. ' because LaTeX numbers sections itself."""
    pat = r"^" + r"\d+\.\s*" * level
    return re.sub(pat, "", text).strip()


def strip_caption(text: str) -> str:
    """Strip the table or figure number prefix because LaTeX numbers captions itself."""
    return re.sub(r"^(Bảng|Hình)\s+\d+\.\d+\.\s*", "", text).strip()


def _list(items, env: str) -> str:
    inner = "\n".join(r"\item " + tex(x) for x in items)
    return f"\\begin{{{env}}}\n{inner}\n\\end{{{env}}}"


def _table(headers, rows, caption=None) -> str:
    ncol = len(headers)
    spec = "|" + "|".join(["l"] + ["c"] * (ncol - 1)) + "|"
    size = r"\footnotesize" if ncol >= 5 else r"\small"
    lines = [r"\begin{table}[H]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + tex(strip_caption(caption)) + "}")
    lines.append(r"\begin{center}")
    lines.append(size)
    lines.append(r"\begin{tabular}{" + spec + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(r"\textbf{" + tex(h) + "}" for h in headers) + r" \\ \hline")
    for row in rows:
        lines.append(" & ".join(tex(str(c)) for c in row) + r" \\ \hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build() -> str:
    """Walk `BLOCKS`, the Word source, and emit the matching LaTeX."""
    chunks = [PREAMBLE]
    skip_next_list = False
    pending_caption = None

    for kind, payload in BLOCKS:
        if kind in ("cover", "center"):
            continue
        if kind == "toc":
            continue
        if kind == "h1":
            t = payload
            if t == "LỜI CẢM ƠN":
                chunks.append(r"\chapter*{LỜI CẢM ƠN}")
                chunks.append(r"\addcontentsline{toc}{chapter}{Lời cảm ơn}")
            elif t == "MỤC LỤC":
                chunks.append(r"\tableofcontents")
            elif t == "DANH MỤC CHỮ VIẾT TẮT":
                chunks.append(r"\chapter*{DANH MỤC CHỮ VIẾT TẮT}")
                chunks.append(r"\addcontentsline{toc}{chapter}{Danh mục chữ viết tắt}")
            elif t == "DANH SÁCH BẢNG":
                chunks.append(r"\phantomsection\addcontentsline{toc}{chapter}{Danh sách bảng}")
                chunks.append(r"\listoftables")
                skip_next_list = True
            elif t == "DANH SÁCH BIỂU ĐỒ":
                chunks.append(r"\phantomsection\addcontentsline{toc}{chapter}{Danh sách biểu đồ}")
                chunks.append(r"\listoffigures")
                skip_next_list = True
            elif t == "TÀI LIỆU THAM KHẢO":
                chunks.append(REFS.replace(
                    r"\chapter*{TÀI LIỆU THAM KHẢO}",
                    "\\chapter*{TÀI LIỆU THAM KHẢO}\n"
                    "\\addcontentsline{toc}{chapter}{Tài liệu tham khảo}"))
                skip_next_list = True
            elif t.startswith("CHƯƠNG"):
                title = t.split(". ", 1)[1] if ". " in t else t
                chunks.append("\\chapter{" + title + "}")
            else:
                chunks.append("\\chapter*{" + t + "}")
            continue
        if kind == "h2":
            chunks.append("\\section{" + strip_num(payload, 2) + "}")
            continue
        if kind == "h3":
            chunks.append("\\subsection{" + strip_num(payload, 3) + "}")
            continue
        if kind == "p":
            chunks.append(tex(payload))
            continue
        if kind == "bullet":
            if skip_next_list:
                skip_next_list = False
                continue
            chunks.append(_list(payload, "itemize"))
            continue
        if kind == "num":
            chunks.append(_list(payload, "enumerate"))
            continue
        if kind == "caption":
            pending_caption = payload
            continue
        if kind == "table":
            headers, rows = payload
            chunks.append(_table(headers, rows, pending_caption))
            pending_caption = None
            continue
        if kind == "img":
            rel, cap = payload
            chunks.append(
                "\\begin{figure}[H]\n\\centering\n"
                "\\includegraphics[width=0.8\\textwidth]{figures/" + Path(rel).name + "}\n"
                "\\caption{" + tex(strip_caption(cap)) + "}\n\\end{figure}")
            continue
        if kind == "pb":
            chunks.append(r"\clearpage")
            continue

    chunks.append("\\end{document}")
    return "\n\n".join(chunks)


def main() -> int:
    """Write `main.tex` and report the block, table and figure counts."""
    target = OUT / "main.tex"
    target.write_text(build(), encoding="utf-8")
    n_tab = sum(1 for b in BLOCKS if b[0] == "table")
    n_img = sum(1 for b in BLOCKS if b[0] == "img")
    print(f"Wrote {target}  ({target.stat().st_size/1024:.1f} KB, "
          f"{len(BLOCKS)} blocks, {n_tab} tables, {n_img} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



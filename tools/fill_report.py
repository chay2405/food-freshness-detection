"""Fill the run-dependent values of the project report (.docx) from the notebook outputs.

Usage:
    python tools/fill_report.py --outputs <folder with metrics.json + png files> \
                                --docx <input report.docx> [--out <output.docx>]

<folder> is the unzipped project_outputs.zip downloaded from Colab. It must contain
metrics.json, training_curves.png, confusion_matrix.png and roc_curve.png.
Needs: pip install python-docx
Nothing is invented: if a value is missing from metrics.json the cell is left unchanged.
"""
import argparse, json, os, re, sys
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def set_cell(cell, text):
    """Replace the text of a cell but keep the formatting of its first run."""
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def pct(x, d=2):
    return f"{x * 100:.{d}f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = json.load(open(os.path.join(a.outputs, "metrics.json")))
    doc = Document(a.docx)
    t_split, t_metrics, t_fruit = doc.tables[0], doc.tables[1], doc.tables[2]

    # Table 1: split counts
    sc = m.get("split_counts")
    if sc:
        for row, key in zip(t_split.rows[1:], ["train", "validation", "test"]):
            set_cell(row.cells[1], f"{sc[key]['images']:,}")
            set_cell(row.cells[2], f"{sc[key]['fresh']:,}")
            set_cell(row.cells[3], f"{sc[key]['rotten']:,}")

    # Table 2: metrics
    vals = {
        "Accuracy": f"{m['accuracy']:.4f}  ({pct(m['accuracy'])})",
        "Precision (Rotten)": f"{m['precision_rotten']:.4f}  ({pct(m['precision_rotten'])})",
        "Recall (Rotten)": f"{m['recall_rotten']:.4f}  ({pct(m['recall_rotten'])})",
        "F1-score (Rotten)": f"{m['f1_rotten']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
    }
    cm = m.get("confusion_matrix")
    if cm:
        vals.update({"True negatives": str(cm[0][0]), "False positives": str(cm[0][1]),
                     "False negatives": str(cm[1][0]), "True positives": str(cm[1][1])})
    for row in t_metrics.rows[1:]:
        label = row.cells[0].text.strip()
        for k, v in vals.items():
            if label.startswith(k):
                set_cell(row.cells[1], v)

    # Table 3: per-fruit
    pf = m.get("per_fruit", {})
    for row in t_fruit.rows[1:]:
        fr = row.cells[0].text.strip().lower()
        if fr in pf:
            set_cell(row.cells[1], str(pf[fr]["test_images"]))
            set_cell(row.cells[2], pct(pf[fr]["accuracy"], 1))

    # Summary paragraph + figures
    n = m.get("test_images", "")
    summary = (f"On the {n} unseen test images the model achieved an accuracy of {pct(m['accuracy'])}, "
               f"with precision {m['precision_rotten']:.4f}, recall {m['recall_rotten']:.4f}, "
               f"F1-score {m['f1_rotten']:.4f} and ROC-AUC {m['roc_auc']:.4f} for the Rotten class.")
    if cm:
        summary += (f" The confusion matrix shows {cm[0][1]} fresh fruit incorrectly flagged as rotten "
                    f"and {cm[1][0]} rotten fruit incorrectly classified as fresh.")
    figs = {"training_curves.png": 6.0, "confusion_matrix.png": 4.2, "roc_curve.png": 3.8}
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt.startswith("The final performance of the model must be reported"):
            p._element.getparent().remove(p._element)          # instruction note, no longer needed
        elif txt.startswith("[Results summary will be inserted"):
            for r in p.runs[1:]:
                r.text = ""
            if p.runs:
                p.runs[0].text = summary
                p.runs[0].italic = False
            else:
                p.add_run(summary)
        else:
            mm = re.match(r"\[Insert figure: (\w+\.png)\]", txt)
            if mm and mm.group(1) in figs:
                img = os.path.join(a.outputs, mm.group(1))
                if os.path.exists(img):
                    for r in p.runs:
                        r.text = ""
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    (p.runs[0] if p.runs else p.add_run()).add_picture(img, width=Inches(figs[mm.group(1)]))
                else:
                    print("WARNING: missing", img, file=sys.stderr)

    out = a.out or re.sub(r"\.docx$", "_filled.docx", a.docx)
    doc.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()

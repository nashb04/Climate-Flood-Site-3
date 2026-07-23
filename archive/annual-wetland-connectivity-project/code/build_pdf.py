"""Generate a short PDF explaining the travel-time-weighted wetland (W) workflow
and its exact parameters — answering the three questions from the chat."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

F="/Users/jared/miniforge3/envs/wetland/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/"
pdfmetrics.registerFont(TTFont("DJ", F+"DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJB", F+"DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJI", F+"DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJB", italic="DJI")

NAVY=colors.HexColor("#0A2A3B"); TEAL=colors.HexColor("#1C7293")
GREEN=colors.HexColor("#2E7D32"); INK=colors.HexColor("#1E293B")
MUT=colors.HexColor("#5b6b75"); PANEL=colors.HexColor("#EEF3F6")

def S(name,**kw):
    base=dict(fontName="DJ",textColor=INK,fontSize=10.5,leading=15)
    base.update(kw); return ParagraphStyle(name,**base)
H1=S("H1",fontName="DJB",fontSize=18,textColor=NAVY,leading=22,spaceAfter=2)
SUB=S("SUB",fontName="DJI",fontSize=10.5,textColor=MUT,spaceAfter=10)
H2=S("H2",fontName="DJB",fontSize=13,textColor=TEAL,spaceBefore=12,spaceAfter=5)
BODY=S("BODY",spaceAfter=6)
Q=S("Q",fontName="DJB",fontSize=11,textColor=NAVY,spaceBefore=6,spaceAfter=2)
A=S("A",leftIndent=12,spaceAfter=4)
FORM=S("FORM",fontName="DJB",fontSize=11,textColor=GREEN,alignment=1,spaceBefore=4,spaceAfter=6)
CAP=S("CAP",fontName="DJI",fontSize=8.5,textColor=MUT)
TH=S("TH",fontName="DJB",fontSize=9,textColor=colors.white)
TC=S("TC",fontSize=9,leading=12)
TCb=S("TCb",fontName="DJB",fontSize=9,leading=12,textColor=NAVY)

st=[]
st.append(Paragraph("Travel-Time–Weighted Wetland Metric (W)", H1))
st.append(Paragraph("Workflow &amp; chosen parameters — Milwaukee River basin pipeline", SUB))
st.append(HRFlowable(width="100%", color=TEAL, thickness=1.4, spaceAfter=8))
st.append(Paragraph("<b>Idea.</b> Each upstream wetland cell is weighted by how hydrologically "
  "connected it is to the gauge — i.e. by its <b>travel time</b> T to the outlet. "
  "Near-stream (short-T) wetlands count more. W is the <b>connectivity-weighted wetland "
  "fraction</b> of the catchment.", BODY))
st.append(Paragraph("W = ( Σ<sub>wetland cells</sub> e<super>−T/τ</super> )  ÷  "
  "( Σ<sub>all catchment cells</sub> e<super>−T/τ</super> )   ∈ [0, 1]", FORM))

# ── Q&A ──────────────────────────────────────────────────────────────────────
st.append(Paragraph("Answers to the three questions", H2))

st.append(Paragraph("Q1 — “Did you use just 1/t, and how did you handle the flat terrain that "
  "made TOC too long?”", Q))
st.append(Paragraph("<b>Kernel.</b> The primary weight is the <b>exponential</b> f(T)=e<super>−T/τ</super>, "
  "not 1/T. We also compute the inverse-time 1/T variant; the two give a very similar ranking, "
  "so results are robust to the choice.", A))
st.append(Paragraph("<b>Flat terrain / over-long TOC.</b> Four safeguards:", A))
st.append(Paragraph("(1) <b>WhiteBox least-cost breaching</b> "
  "(BreachDepressionsLeastCost, dist=2000, fill=True) — not plain depression-filling — to repair "
  "the fragmented drainage on this flat glacial basin (max flow accumulation went from 108 to "
  "1,759 km² on an ~1,809 km² basin once breached).", A))
st.append(Paragraph("(2) <b>resolve_flats</b> adds a tiny synthetic gradient so D8 directions are "
  "defined on flats.  (3) The per-step <b>slope is floored at S<sub>min</sub>=1×10<super>−3</super> "
  "m/m</b>, so velocity ≠ 0 and the time-of-concentration stays finite.", A))
st.append(Paragraph("(4) Because absolute T is still inflated on flat ground (hundreds of hours), "
  "the decay scale <b>τ is set adaptively to each catchment’s median T</b> (see Q3), which "
  "normalises the magnitude away. <i>Known limitation:</i> a few headwater catchments have almost "
  "no relief, so their internal travel-time structure is weak — flagged for sensitivity work.", A))

st.append(Paragraph("Q2 — “How did you turn it into a percentage after summing the weights?”", Q))
st.append(Paragraph("W is a <b>ratio of two summed weights</b>: the numerator sums the kernel weight "
  "e<super>−T/τ</super> over <b>wetland</b> cells; the denominator sums the <i>same</i> weight over "
  "<b>all</b> cells in the catchment. Their ratio is a dimensionless fraction in [0, 1] (cell area "
  "cancels). If the weight were constant it reduces exactly to the plain wetland-area fraction; with "
  "the kernel, near-stream wetland gets a larger share. We report it as W<sub>frac,exp</sub> "
  "(exponential) and W<sub>frac,inv</sub> (inverse-time).", A))

st.append(Paragraph("Q3 — “What was your decay parameter?”", Q))
st.append(Paragraph("<b>τ = the catchment’s median travel time</b> — adaptive, one value per basin, "
  "<i>not</i> a fixed constant. We first tried a fixed τ = 24 h, but on this flat terrain travel times "
  "are hundreds of hours, so e<super>−T/24h</super> ≈ 0 for almost every cell (degenerate). Using the "
  "median-T makes the weighting well-scaled in every basin. As a kernel-free check we also report a "
  "<b>near-stream fraction</b> = the share of wetland whose T is below the 33rd percentile of the "
  "catchment’s travel times.", A))

# ── parameter table ──────────────────────────────────────────────────────────
st.append(Paragraph("Full parameter list", H2))
def r(a,b,c): return [Paragraph(a,TCb),Paragraph(b,TC),Paragraph(c,TC)]
data=[[Paragraph("Step / parameter",TH),Paragraph("Value",TH),Paragraph("Note",TH)],
 r("DEM","10 m, USGS 3DEP","reprojected to UTM 16N"),
 r("Wetland classes","LCMAP “wetland” (class 6); NLCD 90+95","annual land cover"),
 r("DEM conditioning","WhiteBox BreachDepressionsLeastCost (dist=2000, fill=True) + resolve_flats","flat-terrain fix"),
 r("Flow routing","D8 (pysheds)","single-direction"),
 r("Stream threshold","flow accumulation &gt; 5,000 cells (= 0.5 km²)","defines channels"),
 r("Slope floor S<sub>min</sub>","1×10<super>−3</super> m/m","keeps velocity &gt; 0 on flats"),
 r("Overland velocity","TR-55: V = K·√S,  K = 4.92 m·s<super>−1</super>","shallow concentrated flow"),
 r("Channel velocity","Manning: V = (1/n)·R<super>2/3</super>·√S,  n = 0.04, R = 0.7 m","uniform hydraulics"),
 r("Per-cell time","L / V   (L = D8 step length)","seconds"),
 r("Total travel time T","Σ per-cell time along D8 path to gauge","pysheds distance_to_outlet"),
 r("Kernel f(T)","e<super>−T/τ</super> (primary);  1/T (secondary)","near-stream weighted higher"),
 r("Decay τ","catchment <b>median</b> T  (adaptive)","per basin; not fixed"),
 r("Metric W","Σ<sub>wet</sub> f(T) ÷ Σ<sub>all</sub> f(T)  ∈ [0,1]","weighted wetland fraction"),
 r("Robustness","near-stream fraction (T ≤ 33rd pctile)","kernel-free check"),
]
t=Table(data,colWidths=[1.55*inch,3.0*inch,2.05*inch])
t.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),TEAL),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,PANEL]),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cdd9df")),
 ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
 ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
st.append(t)
st.append(Spacer(1,8))
st.append(Paragraph("Implementation: pysheds (routing), WhiteBox (conditioning), py3dep (DEM), "
  "Planetary-Computer/LCMAP and NLCD (wetland). W is computed in build_traveltime_W.py.", CAP))

SimpleDocTemplate("/Users/jared/Wetland/Weighted_Wetland_Workflow.pdf", pagesize=letter,
  topMargin=0.7*inch, bottomMargin=0.7*inch, leftMargin=0.8*inch, rightMargin=0.8*inch
).build(st)
print("Wrote /Users/jared/Wetland/Weighted_Wetland_Workflow.pdf")

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

game_model=BayesianNetwork([
    ("P1draw","P2draw"),
    ("P1draw","P1call"),
    ("P2draw","P2call"),
    ("P1call2","P2call"),
    ("P1call2","P1call"),
    ("P1draw","win"),
    ("P2draw","win"),
    ("P1call","earn"),
    ("P2call","earn"),
])
cpd_P1draw = TabularCPD(
    variable="P1draw", variable_card=5, values=[[0.2], [0.2], [0.2] ,[0.2], [0.2]]
)
cpd_P2draw = TabularCPD(
    variable="P2draw",
    variable_card=5,
    values=[[0, 0.25, 0.25 ,0.25, 0.25],
            [0.25, 0, 0.25 ,0.25, 0.25],
            [0.25, 0.25, 0 ,0.25, 0.25],
            [0.25, 0.25, 0.25 ,0, 0.25],
            [0.25, 0.25, 0.25 ,0.25, 0]],
    evidence=["P1draw"],
    evidence_card=[5]
)
cpd_P1call = TabularCPD(
    variable="P2call",
    variable_card=2,
    values=[[0.95, 0.75, 0.6, 0.3, 0.03],[0.05,0.25,0.4,0.7,0.97]],
    evidence=["P1draw"],
    evidence_card=[5]
)

cpd_P2call = TabularCPD(
    variable="P2call",
    variable_card=2,
    values=[[1, 1, 0.85, 0.8, 0.5,0.45, 0.2,0.15,0,0],[0,0,0.15,0.2,0.5,0.55,0.8,0.85,1,1]],
    evidence=["P2draw","P1call"],
    evidence_card=[5,2]
)
cpd_P1call2 = TabularCPD(
    variable="P2call2",
    variable_card=2,
    values=[[1, 0, 1, 0, 0.85, 0, 0.8, 0, 0.5, 0,0.45, 0, 0.2, 0,0.15, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.15, 0, 0.2, 0, 0.5, 0,0.55, 0, 0.8, 0,0.85, 0, 1, 0, 1, 0]],
    evidence=["P1draw","P2call","P1call"],
    evidence_card=[5,2,2]
)


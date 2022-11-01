from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.inference import CausalInference
def make_decision(e0,e1,e2,e3,e4):
    return -2 * e0 -e1 +e3 +2*e4   # expected gain pentru jucatorul 1
game_model=BayesianNetwork([
    ("P1draw","P2draw"),
    ("P1draw","P1call"),
    ("P2draw","P2call"),
    ("P1call","P2call"),
    ("P1call","P1call2"),
    ("P2call","P1call2"),
    ("P1draw","P1call2"),
    ("P1draw","win"),
    ("P2draw","win"),
    ("P1calldef","earn"),
    ("P2call","earn"),
    ("win",'earn'),
    ("P1call","P1calldef"),
    ("P1call2","P1calldef"),
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
    variable="P1call",
    variable_card=2,
    values=[[0.95, 0.75, 0.6, 0.3, 0.03],[0.05,0.25,0.4,0.7,0.97]],
    #inversate pt a schimba raspunsul la pct b,
    # [0.95, 0.75, 0.6, 0.3, 0.03],[0.05,0.25,0.4,0.7,0.97]]
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
    variable="P1call2",
    variable_card=2,
    values=[[1, 1, 1, 1, 0.85, 1, 0.8, 1, 0.5, 1,0.45, 1, 0.2, 1,0.15, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0.15, 0, 0.2, 0, 0.5, 0,0.55, 0, 0.8, 0,0.85, 0, 1, 0, 1, 0]],
    #inversate pentru a schimba raspunsul pe la pct b
    evidence=["P1draw","P2call","P1call"],
    evidence_card=[5,2,2]
)
cpd_P1calldef=TabularCPD(
    variable="P1calldef",
    variable_card=2,
    values=[[1,0,0,0],[0,1,1,1]], # daca jucatorul 1 a pariat cel putin o data din cele 2 oportunitati
    evidence=["P1call","P1call2"],
    evidence_card=[2,2]
)
cpd_win = TabularCPD(
    variable="win",
    variable_card=2,
    values=[[0,0,0,0,0,
            1,0,0,0,0,
            1,1,0,0,0,
            1,1,1,0,0,
            1,1,1,1,0],
            [1,1,1,1,1,
             0,1,1,1,1,
             0,0,1,1,1,
             0,0,0,1,1,
             0,0,0,0,1]],
    evidence=["P1draw","P2draw"],
    evidence_card=[5,5]
)
cpd_earn=TabularCPD(
    variable="earn",
    variable_card=5,
    values=[[0,0,0,0,0,0,0,1],[0,1,0,0,0,1,0,0],[1,0,0,0,1,0,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,0]],                           #0= -2$ 1=-1$ 2=0$ 3=1$ 4=2$ pt jucatorul 1
    evidence=["win","P1calldef","P2call"],
    evidence_card=[2, 2, 2]
)
game_model.add_cpds(cpd_P1draw,cpd_P2draw,cpd_P1call,cpd_P2call,cpd_P1call2,cpd_win,cpd_earn,cpd_P1calldef)
infer=VariableElimination(game_model)
q1=infer.query(variables=["earn"], evidence={"P1draw" : 3 , "P1calldef": 1})
q2=infer.query(variables=["earn"], evidence={"P1draw" : 3 , "P1calldef": 0})
print(make_decision(q1.values[0],q1.values[1],q1.values[2],q1.values[3],q1.values[4]) -
      make_decision(q2.values[0],q2.values[1],q2.values[2],q2.values[3],q2.values[4])) # >0 deci prima varianta, de a paria avantajeaza jucatorul 1
q1=infer.query(variables=["earn"], evidence={"P2draw" : 2 , "P1calldef": 1, "P2call" : 1})
q2=infer.query(variables=["earn"], evidence={"P2draw" : 2 , "P1calldef": 1, "P2call" : 0})
print(make_decision(q1.values[0],q1.values[1],q1.values[2],q1.values[3],q1.values[4]) -
      make_decision(q2.values[0],q2.values[1],q2.values[2],q2.values[3],q2.values[4])) # >0 deci jucatorul 2 ar trebui sa iasa din joc
# Pentru ca jucatorul 2 sa iasa in avantaj cand pariaza ( la pct b) putem inversa strategia jucatorului 1 (sa parieze mai des cand are carti mai slabe,
#adica 1-P(call) pentru fiecare decizie a jucatorului 1 fata de cum sunt setate acum, inferenta da rezultatul de mai jos <0:
print(make_decision(0.8192,0,0,0,0.1808) - make_decision(0,0,0,1,0))
#Nu am gasit o instanta in care jucatorul 1 este sfatuit sa renunte cu un rege de frunza

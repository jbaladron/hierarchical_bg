from ANNarchy import * 
import pylab as plt
import random
import sys
import scipy.spatial.distance

exp=int(sys.argv[1])

#General networks parameters
baseline_dopa = 0.1



#Neuron models

LinearNeuron = Neuron(
    parameters= """
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        lesion = 1.0

    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
    """
)



LinearNeuron_trace = Neuron(
    parameters= """
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        tau_trace = 120.0
        lesion = 1.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
        tau_trace*dtrace/dt + trace = r
    """
)

DopamineNeuron = Neuron(
    parameters="""
        tau = 10.0
        firing = 0
        inhibition = 0.0
        baseline = 0.0
        exc_threshold = 0.0
        factor_inh = 10.0
    """,
    equations="""
        ex_in = if (sum(exc)>exc_threshold): 1 else: 0
        s_inh = sum(inh)
        aux = if (firing>0): (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh)+baseline)  else: baseline
        tau*dmp/dt + mp =  aux
        r = if (mp>0.0): mp else: 0.0
    """
)

InputNeuron = Neuron(
    parameters="""
        tau = 1.5
        baseline = 0.0
    """,
    equations="""
        tau*dmp/dt + mp = baseline
	r = if (mp>0.0): mp else: 0.0
    """


)



###################################################################################################################################################
###################################################################################################################################################

#Synapse models
PostCovariance = Synapse(
    parameters="""
        tau = 1000.0
        tau_alpha = 10.0 
        regularization_threshold = 1.0
        threshold_post = 0.0
        threshold_pre = 0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) 


        trace = (pre.r - mean(pre.r) - threshold_pre) * pos(post.r - mean(post.r) - threshold_post)
	delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post)*w)
        tau*dw/dt = delta : min=0
   """
)
PreCovariance = Synapse(
    parameters="""
        tau = 1000.0
        tau_alpha = 10.0 
        regularization_threshold = 1.0
        threshold_post = 0.0
        threshold_pre = 0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) 


        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
	delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post)*w)
        tau*dw/dt = delta : min=0
   """
)

ReversedSynapse = Synapse(
    parameters="""
        reversal = 0.3
    """,
    psp="""
        w*pos(reversal-pre.r)
    """    

)

#DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
DAPostCovarianceNoThreshold = Synapse(
    parameters="""
        tau=1000.0
        tau_alpha=10.0 
        regularization_threshold=1.0 
        baseline_dopa = 0.1
        K_burst = 1.0
        K_dip = 0.4
        DA_type = 1 
        threshold_pre=0.0
        threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold) 
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)

	condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum

        

        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min=0 
    """


)


DAPostCovarianceNoThreshold_trace = Synapse(
    parameters="""
        tau=1000.0
        tau_alpha=10.0 
        regularization_threshold=1.0 
        baseline_dopa = 0.1
        K_burst = 1.0
        K_dip = 0.4
        DA_type = 1 
        threshold_pre=0.0
        threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold) 
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)

        trace = pos(post.trace -  mean(post.trace) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)

	condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum

        

        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min=0 
    """


)


#Excitatory synapses STN -> SNr
DAPreCovariance_excitatory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0 
    regularization_threshold=1.0 
    baseline_dopa = 0.1  
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    """,
    equations = """
        tau_alpha*dalpha/dt  = pos( post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
        aux = if (trace<0.0): 1 else: 0
        dopa_mod = if (dopa_sum>0): K_burst * dopa_sum else: K_dip * dopa_sum * aux
        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min=0 

        
    """

)


#Inhibitory synapses SNr -> SNr and STRD2 -> GPe
DAPreCovariance_inhibitory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0 
    regularization_threshold=1.0 
    baseline_dopa = 0.1    
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    neg = 1
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): neg else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        trace2 = trace

        delta = dopa_mod * trace2 - alpha * pos(trace2)
        tau*dw/dt = delta : min=0 
    """


)


DAPreCovariance_inhibitory_trace = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0 
    regularization_threshold=1.0 
    baseline_dopa = 0.1 
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.trace) - post.trace  - threshold_post)
        aux = if (trace>0): 1 else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        trace2 = trace

        delta = dopa_mod * trace2 - alpha * pos(trace2)
        tau*dw/dt = delta : min=0 
    """


)


DAPrediction = Synapse(
    parameters="""
        tau = 100000.0
        baseline_dopa = 0.1
   """,
   equations="""
       aux = if (post.sum(exc)>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
       tau*dw/dt = delta : min=0 
   """


)



###################################################################################################################################################
###################################################################################################################################################

#CORTICAL NEURONS

#these are the input cells that represents the stimulus (door and audio)
Input_neurons = Population(name='Input',geometry=40,neuron=InputNeuron)

Context = Population(name='Context',geometry=2,neuron=InputNeuron)

#motor cortex, represents the two actions
PM = Population(name="PM", geometry = 2, neuron=LinearNeuron)
PM.tau = 30.0
PM.noise = 0.01

#Neurons to represent the state of the rat
Propio = Population(name="Propio", geometry=3, neuron=LinearNeuron)
Propio.tau = 10.0
Propio.noise = 0.01

#Neurons to represent the different objectives
Objectives = Population(name="Objectives", geometry=3, neuron=LinearNeuron)
Objectives.tau = 40.0
Objectives.noise = 0.01

Objectives_extra = Population(name="Objectives", geometry=15, neuron=LinearNeuron)
Objectives_extra.tau = 40.0
Objectives_extra.noise = 0.01


#Shortcut
IL = Population(name='IL',geometry=2,neuron=LinearNeuron)
IL.tau = 10
IL.baseline = 0.0
IL.noise = 0.1

#dorsomedial loop

# Striatum direct pathway (each group receibes input from one goal/reward)
StrD1_caudate0 = Population(name="StrD1_caudate0", geometry=(2,2),neuron = LinearNeuron)
StrD1_caudate0.tau = 10.0
StrD1_caudate0.noise = 0.3 
StrD1_caudate0.baseline = 0.0
StrD1_caudate1 = Population(name="StrD1_caudate1", geometry=(2,2),neuron = LinearNeuron)
StrD1_caudate1.tau = 10.0
StrD1_caudate1.noise = 0.3 #
StrD1_caudate1.baseline = 0.0



# Striatum indirect pathway
StrD2_caudate0 = Population(name="StrD2_caudate0", geometry = (3,3), neuron=LinearNeuron)
StrD2_caudate0.tau = 10.0
StrD2_caudate0.noise = 0.01
StrD2_caudate0.baseline = 0.0
StrD2_caudate1 = Population(name="StrD2_caudate1", geometry = (3,3), neuron=LinearNeuron)
StrD2_caudate1.tau = 10.0
StrD2_caudate1.noise = 0.01
StrD2_caudate1.baseline = 0.0

# Striatum feedback pathway
StrThal_caudate = Population(name="StrThal_caudate", geometry = 2, neuron=LinearNeuron)
StrThal_caudate.tau = 10.0
StrThal_caudate.noise = 0.01
StrThal_caudate.baseline = 0.4

# SNr
SNr_caudate = Population(name="SNr_caudate", geometry = 2, neuron=LinearNeuron_trace)
SNr_caudate.tau = 10.0
SNr_caudate.noise = 0.3
SNr_caudate.baseline = 1.5 
SNr_caudate.tau_trace = 200.

# STN
STN_caudate0 = Population(name="STN_caudate0", geometry = (4,4), neuron=LinearNeuron)
STN_caudate0.tau = 10.0
STN_caudate0.noise = 0.01
STN_caudate0.baseline = 0.0
STN_caudate1 = Population(name="STN_caudate1", geometry = (4,4), neuron=LinearNeuron)
STN_caudate1.tau = 10.0
STN_caudate1.noise = 0.01
STN_caudate1.baseline = 0.0

# GPe
GPe_caudate = Population(name="GPe_caudate", geometry = 2, neuron=LinearNeuron)
GPe_caudate.tau = 10.0
GPe_caudate.noise = 0.05
GPe_caudate.baseline = 1.0

# Thalamus
VA_caudate = Population(name="VA_caudate", geometry=2, neuron=LinearNeuron)
VA_caudate.tau = 10.0
VA_caudate.noise = 0.05
VA_caudate.baseline = 0.0

# PFC (introduce baseline in the thalamus of the first loop)
PFC_caudate = Population(name="PFC_caudate", geometry=2, neuron=LinearNeuron)
PFC_caudate.tau = 10.0
PFC_caudate.noise = 0.05
PFC_caudate.baselie = 0.0

#MOTOR LOOP

# Striatum direct pathway
StrD1_putamen = Population(name="StrD1_putamen", geometry=(2),neuron = LinearNeuron_trace)
StrD1_putamen.tau = 10.0 
StrD1_putamen.noise = 0.1/2. 
StrD1_putamen.baseline = 0.0

# Striatum indirect pathway
StrD2_putamen = Population(name="StrD2_putamen", geometry = (2,2), neuron=LinearNeuron)
StrD2_putamen.tau = 10.0
StrD2_putamen.noise = 0.1/2.
StrD2_putamen.baseline = 0.0

# Striatum feedback pathway
StrThal_putamen = Population(name="StrThal_putamen", geometry = 2, neuron=LinearNeuron)
StrThal_putamen.tau = 5.0
StrThal_putamen.noise = 0.01
StrThal_putamen.baseline = 0.4

# SNr
SNr_putamen = Population(name="SNr_putamen", geometry =2, neuron=LinearNeuron_trace)
SNr_putamen.tau = 5.0 
SNr_putamen.noise = 0.005 
SNr_putamen.baseline = 1.1 
SNr_putamen.tau_trace = 200.

# STN
STN_putamen = Population(name="STN_putamen", geometry = (2,2), neuron=LinearNeuron)
STN_putamen.tau = 10.0
STN_putamen.noise = 0.01
STN_putamen.baseline = 0.0

# GPe
GPe_putamen = Population(name="GPe_putamen", geometry = 2, neuron=LinearNeuron)
GPe_putamen.tau = 10.0
GPe_putamen.noise = 0.001
GPe_putamen.baseline = 1.0

# VA	
VA_putamen = Population(name="VA_putamen", geometry=2, neuron=LinearNeuron)
VA_putamen.tau = 8.0
VA_putamen.noise = 0.0
VA_putamen.baseline = 0.0


#REWARD
SNc_put = Population(name='SNc_put',geometry=2,neuron=DopamineNeuron)
SNc_put.exc_threshold=1.5
SNc_put.baseline = baseline_dopa
SNc_put.factor_inh = 1.0

SNc_caud = Population(name='SNc_cau',geometry=2,neuron=DopamineNeuron)
SNc_caud.baseline = baseline_dopa

#These cells are activated when each of the rewards are received.
PPTN = Population(name="PPTN", geometry=2, neuron=InputNeuron)
PPTN.tau = 1.0

#This is used for the Packard task
Hippo = Population(name='Hippocampus',geometry=1,neuron=LinearNeuron)
Hippo.tau = 30
Hippo.noise = 0.0
Hippo.baseline =0.0



####################################################################################################################################################
####################################################################################################################################################

#SYNAPSES

#Associative loop

#Baseline for the thalamus of the first loop
VAPFC_11 = Projection(pre=VA_caudate[0],post=PFC_caudate[0],target='exc')
VAPFC_11.connect_all_to_all(weights=1.0)
VAPFC_22 = Projection(pre=VA_caudate[1],post=PFC_caudate[1],target='exc')
VAPFC_22.connect_all_to_all(weights=1.0)

PFCVA_11 = Projection(pre=PFC_caudate[0],post=VA_caudate[0],target="exc")
PFCVA_11.connect_all_to_all(weights = 0.35) #0.15
PFCVA_22 = Projection(pre=PFC_caudate[1],post=VA_caudate[1],target="exc")
PFCVA_22.connect_all_to_all(weights = 0.35)

ITPFC = Projection(pre=Input_neurons,post=PFC_caudate,target="exc")#,synapse=PostCovariance)
ITPFC.connect_all_to_all( weights = Uniform(0.2,0.3)) 


#Input (door and audio) to striatum
ITStrD1_caudate0 = Projection(pre=Input_neurons,post=StrD1_caudate0,target='exc',synapse=DAPostCovarianceNoThreshold)
ITStrD1_caudate0.connect_all_to_all(weights = Normal(0.1,0.02))  
ITStrD1_caudate0.tau = 100  
ITStrD1_caudate0.regularization_threshold = 1.0
ITStrD1_caudate0.tau_alpha = 2.0
ITStrD1_caudate0.baseline_dopa = baseline_dopa
ITStrD1_caudate0.K_dip = 0.05
ITStrD1_caudate0.K_burst = 1.0
ITStrD1_caudate0.DA_type = 1
ITStrD1_caudate0.threshold_pre = 0.35 
ITStrD1_caudate0.threshold_post = 0.0

ITStrD1_caudate1 = Projection(pre=Input_neurons,post=StrD1_caudate1,target='exc',synapse=DAPostCovarianceNoThreshold)
ITStrD1_caudate1.connect_all_to_all(weights = Normal(0.1,0.02)) 
ITStrD1_caudate1.tau = 100  
ITStrD1_caudate1.regularization_threshold =  1.0
ITStrD1_caudate1.tau_alpha = 2.0
ITStrD1_caudate1.baseline_dopa = baseline_dopa
ITStrD1_caudate1.K_dip = 0.05
ITStrD1_caudate1.K_burst = 1.0
ITStrD1_caudate1.DA_type = 1
ITStrD1_caudate1.threshold_pre = 0.35 
ITStrD1_caudate1.threshold_post = 0.0

#Connection between the reward signals to the striatum
ObjStrD1_caudate0 = Projection(pre=Objectives_extra[0:5],post=StrD1_caudate0,target='exc')
ObjStrD1_caudate0.connect_all_to_all(weights=0.2)

ObjStrD1_caudate1 = Projection(pre=Objectives_extra[5:10],post=StrD1_caudate1,target='exc')
ObjStrD1_caudate1.connect_all_to_all(weights=0.2)

ObjStrD1_caudate01 = Projection(pre=Objectives_extra[0:5],post=StrD1_caudate1,target='inh')
ObjStrD1_caudate01.connect_all_to_all(weights=0.6)
ObjStrD1_caudate10 = Projection(pre=Objectives_extra[5:10],post=StrD1_caudate0,target='inh')
ObjStrD1_caudate10.connect_all_to_all(weights=0.6)

#Direct pathway
StrD1SNr_caudate0 = Projection(pre=StrD1_caudate0,post=SNr_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
StrD1SNr_caudate0.connect_all_to_all(weights=Normal(0.2,0.01))
StrD1SNr_caudate0.tau = 550 
StrD1SNr_caudate0.regularization_threshold = 1.5
StrD1SNr_caudate0.tau_alpha = 20.0
StrD1SNr_caudate0.baseline_dopa = 2*baseline_dopa
StrD1SNr_caudate0.K_dip = 0.9
StrD1SNr_caudate0.K_burst = 1.0
StrD1SNr_caudate0.threshold_post = 0.3 
StrD1SNr_caudate0.threshold_pre = 0.15
StrD1SNr_caudate0.DA_type=1
StrD1SNr_caudate0.neg = 5.0

StrD1SNr_caudate1 = Projection(pre=StrD1_caudate1,post=SNr_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
StrD1SNr_caudate1.connect_all_to_all(weights=Normal(0.2,0.01)) 
StrD1SNr_caudate1.tau = 550 
StrD1SNr_caudate1.regularization_threshold = 1.5
StrD1SNr_caudate1.tau_alpha = 20.0
StrD1SNr_caudate1.baseline_dopa = 2*baseline_dopa
StrD1SNr_caudate1.K_dip = 0.9
StrD1SNr_caudate1.K_burst = 1.0
StrD1SNr_caudate1.threshold_post = 0.3 
StrD1SNr_caudate1.threshold_pre = 0.15
StrD1SNr_caudate1.DA_type=1
StrD1SNr_caudate1.neg = 5.0

#Indirect pathway
ITStrD2_caudate0 = Projection(pre=Input_neurons,post=StrD2_caudate0,target='exc',synapse=DAPostCovarianceNoThreshold)
ITStrD2_caudate0.connect_all_to_all(weights = Normal(0.01,0.005)) 
ITStrD2_caudate0.tau = 10.0
ITStrD2_caudate0.regularization_threshold = 1.5
ITStrD2_caudate0.tau_alpha = 1.0
ITStrD2_caudate0.baseline_dopa = baseline_dopa
ITStrD2_caudate0.K_dip = 0.2
ITStrD2_caudate0.K_burst = 1.0
ITStrD2_caudate0.DA_type = -1
ITStrD2_caudate0.threshold_pre = 0.2 
ITStrD2_caudate0.threshold_post = 0.05

ITStrD2_caudate1 = Projection(pre=Input_neurons,post=StrD2_caudate1,target='exc',synapse=DAPostCovarianceNoThreshold)
ITStrD2_caudate1.connect_all_to_all(weights = Normal(0.01,0.005))
ITStrD2_caudate1.tau = 10.0
ITStrD2_caudate1.regularization_threshold = 1.5
ITStrD2_caudate1.tau_alpha = 1.0
ITStrD2_caudate1.baseline_dopa = baseline_dopa
ITStrD2_caudate1.K_dip = 0.2
ITStrD2_caudate1.K_burst = 1.0
ITStrD2_caudate1.DA_type = -1
ITStrD2_caudate1.threshold_pre = 0.2 
ITStrD2_caudate1.threshold_post = 0.05

ObjStrD2_caudate0 = Projection(pre=Objectives_extra[0:5],post=StrD2_caudate0,target='exc')
ObjStrD2_caudate0.connect_all_to_all(weights = 0.2)
ObjStrD2_caudate1 = Projection(pre=Objectives_extra[0:5],post=StrD2_caudate1,target='exc')
ObjStrD2_caudate1.connect_all_to_all(weights = 0.2)


StrD2GPe_caudate0 = Projection(pre=StrD2_caudate0,post=GPe_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
StrD2GPe_caudate0.connect_all_to_all(weights=0.01) 
StrD2GPe_caudate0.tau = 600 #600.0
StrD2GPe_caudate0.regularization_threshold = 10.5
StrD2GPe_caudate0.tau_alpha = 20.0
StrD2GPe_caudate0.baseline_dopa = 2*baseline_dopa
StrD2GPe_caudate0.K_dip = 0.1#0.1
StrD2GPe_caudate0.K_burst = 1.2#1.2
StrD2GPe_caudate0.threshold_post = 0.0
StrD2GPe_caudate0.threshold_pre = 0.2
StrD2GPe_caudate0.DA_type = -1

StrD2GPe_caudate1 = Projection(pre=StrD2_caudate1,post=GPe_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
StrD2GPe_caudate1.connect_all_to_all(weights=0.01) 
StrD2GPe_caudate1.tau = 600 #600.0
StrD2GPe_caudate1.regularization_threshold = 10.5
StrD2GPe_caudate1.tau_alpha = 20.0
StrD2GPe_caudate1.baseline_dopa = 2*baseline_dopa
StrD2GPe_caudate1.K_dip = 0.1#0.1
StrD2GPe_caudate1.K_burst = 1.2#1.2
StrD2GPe_caudate1.threshold_post = 0.0
StrD2GPe_caudate1.threshold_pre = 0.2
StrD2GPe_caudate1.DA_type = -1

GPeSNr_caudate = Projection(pre=GPe_caudate,post=SNr_caudate,target='inh')
GPeSNr_caudate.connect_one_to_one(weights=1.0)

#Hyperdirect pathway
ITSTN_caudate0 = Projection(pre=Input_neurons, post=STN_caudate0, target='exc')#,synapse=DAPostCovarianceNoThreshold)
ITSTN_caudate0.connect_all_to_all(weights = Uniform(0.0,0.001)) 
ITSTN_caudate0.tau = 1500.0 #1000
ITSTN_caudate0.regularization_threshold = 1.0
ITSTN_caudate0.tau_alpha = 1.0
ITSTN_caudate0.baseline_dopa = baseline_dopa
ITSTN_caudate0.K_dip = 0.4#0.4
ITSTN_caudate0.K_burst = 1.0#1.0
ITSTN_caudate0.DA_type = 1
ITSTN_caudate0.threshold_pre = 0.15

ITSTN_caudate1 = Projection(pre=Input_neurons, post=STN_caudate1, target='exc')#,synapse=DAPostCovarianceNoThreshold)
ITSTN_caudate1.connect_all_to_all(weights = Uniform(0.0,0.001)) 
ITSTN_caudate1.tau = 1500.0 #1000
ITSTN_caudate1.regularization_threshold = 1.0
ITSTN_caudate1.tau_alpha = 1.0
ITSTN_caudate1.baseline_dopa = baseline_dopa
ITSTN_caudate1.K_dip = 0.4#0.4
ITSTN_caudate1.K_burst = 1.0#1.0
ITSTN_caudate1.DA_type = 1
ITSTN_caudate1.threshold_pre = 0.15

ObjSTN_caudate0 = Projection(pre=Objectives_extra[0:5], post=STN_caudate0, target='exc')
ObjSTN_caudate0.connect_all_to_all(weights = 0.2)
ObjSTN_caudate1 = Projection(pre=Objectives_extra[5:10], post=STN_caudate1, target='exc',synapse=DAPostCovarianceNoThreshold)
ObjSTN_caudate1.connect_all_to_all(weights = 0.2)


STNSNr_caudate0 = Projection(pre=STN_caudate0,post=SNr_caudate,target='exc')#,synapse=DAPreCovariance_excitatory)
STNSNr_caudate0.connect_all_to_all(weights=Uniform(0.0012,0.0014)) 
STNSNr_caudate0.tau = 9000
STNSNr_caudate0.regularization_threshold = 1.5
STNSNr_caudate0.tau_alpha = 1.0
STNSNr_caudate0.baseline_dopa = 2*baseline_dopa
STNSNr_caudate0.K_dip = 0.4
STNSNr_caudate0.K_burst = 1.0
STNSNr_caudate0.thresholdpost =-0.15
STNSNr_caudate0.DA_type = 1

STNSNr_caudate1 = Projection(pre=STN_caudate1,post=SNr_caudate,target='exc')#,synapse=DAPreCovariance_excitatory)
STNSNr_caudate1.connect_all_to_all(weights=Uniform(0.0012,0.0014)) 
STNSNr_caudate1.tau = 9000
STNSNr_caudate1.regularization_threshold = 1.5
STNSNr_caudate1.tau_alpha = 1.0
STNSNr_caudate1.baseline_dopa = 2*baseline_dopa
STNSNr_caudate1.K_dip = 0.4
STNSNr_caudate1.K_burst = 1.0
STNSNr_caudate1.thresholdpost =-0.15
STNSNr_caudate1.DA_type = 1


#Local inhibition
weight_local_inh = 0.8
StrD1StrD1_caudate0 = Projection(pre=StrD1_caudate0,post=StrD1_caudate0,target='inh')
StrD1StrD1_caudate0.connect_all_to_all(weights = weight_local_inh)
StrD1StrD1_caudate1 = Projection(pre=StrD1_caudate1,post=StrD1_caudate1,target='inh')
StrD1StrD1_caudate1.connect_all_to_all(weights = weight_local_inh)

weight_stn_inh = 0.3
STNSTN_caudate0 = Projection(pre=STN_caudate0,post=STN_caudate0,target='inh')
STNSTN_caudate0.connect_all_to_all(weights = weight_stn_inh)
STNSTN_caudate1 = Projection(pre=STN_caudate1,post=STN_caudate1,target='inh')
STNSTN_caudate1.connect_all_to_all(weights = weight_stn_inh)

PFCPFC_caudate = Projection(pre=PFC_caudate,post = PFC_caudate,target='inh')
PFCPFC_caudate.connect_all_to_all(weights = 0.12)

weight_inh_sd2 = 0.5
StrD2StrD2_caudate0 = Projection(pre=StrD2_caudate0,post=StrD2_caudate0,target='inh')
StrD2StrD2_caudate0.connect_all_to_all(weights=weight_inh_sd2)
StrD2StrD2_caudate1 = Projection(pre=StrD2_caudate1,post=StrD2_caudate1,target='inh')
StrD2StrD2_caudate1.connect_all_to_all(weights=weight_inh_sd2)

StrThalStrThal_caudate = Projection(pre=StrThal_caudate,post=StrThal_caudate,target='inh')
StrThalStrThal_caudate.connect_all_to_all(weights=0.5)

SNrSNr_caudate = Projection(pre=SNr_caudate,post=SNr_caudate,target='exc',synapse=ReversedSynapse)
SNrSNr_caudate.connect_all_to_all(weights=0.8)
SNrSNr_caudate.reversal = 0.4

VAVA_caudate = Projection(pre=VA_caudate,post=VA_caudate,target='inh')
VAVA_caudate.connect_all_to_all(weights=1.1)

#Feedback from the thalamus
ObjStrThal_caudate = Projection(pre=Objectives[1:3],post=StrThal_caudate,target='exc')
ObjStrThal_caudate.connect_one_to_one(weights=1.2)

StrThalGPe_caudate = Projection(pre=StrThal_caudate,post=GPe_caudate,target='inh')
StrThalGPe_caudate.connect_one_to_one(weights=0.3) 

StrThalSNr_caudate = Projection(pre=StrThal_caudate,post=SNr_caudate,target='inh')
StrThalSNr_caudate.connect_one_to_one(weights=1.1) #1.1

#Output connections
SNrVA_caudate = Projection(pre=SNr_caudate,post=VA_caudate,target='inh')
SNrVA_caudate.connect_one_to_one(weights=2.0)

VAObj_caudate = Projection(pre=VA_caudate,post=Objectives[1:3],target='exc')
VAObj_caudate.connect_one_to_one(weights=2.0) #1.6

#Motor loop

#Direct pathway
ObjStrD1_putamen = Projection(pre=Objectives[1:3],post=StrD1_putamen,target='exc',synapse=DAPostCovarianceNoThreshold_trace)
ObjStrD1_putamen.connect_all_to_all(weights = Normal(0.7,0.01))  #Normal(0.25,0.01)) 
ObjStrD1_putamen.tau = 600.0
ObjStrD1_putamen.regularization_threshold = 0.9
ObjStrD1_putamen.tau_alpha = 15
ObjStrD1_putamen.baseline_dopa = 2*baseline_dopa
ObjStrD1_putamen.K_dip = 0.05 
ObjStrD1_putamen.K_burst = 1.2 
ObjStrD1_putamen.DA_type = 1
ObjStrD1_putamen.threshold_pre = 0.1
ObjStrD1_putamen.threshold_post = 0.0#0.4

'''
ContextStrD1_putamen = Projection(pre=Context,post=StrD1_putamen,target='exc',synapse=DAPostCovarianceNoThreshold_trace)
ContextStrD1_putamen.connect_all_to_all(weights = Normal(0.25,0.01)) 
ContextStrD1_putamen.tau = 2000.0 
ContextStrD1_putamen.regularization_threshold = 1.0
ContextStrD1_putamen.tau_alpha = 1
ContextStrD1_putamen.baseline_dopa = 2*baseline_dopa
ContextStrD1_putamen.K_dip = 0.05 
ContextStrD1_putamen.K_burst = 1.2
ContextStrD1_putamen.DA_type = 1
ContextStrD1_putamen.threshold_pre = 0.05
ContextStrD1_putamen.threshold_post = 0.4
'''

weights_snr = np.random.rand(2,2)/2.
#actions = np.random.randint(4,size=16)
actions = [0,1]
actions = np.random.permutation(actions)
weights_snr[actions,range(2)] += 0.6
StrD1SNr_putamen = Projection(pre=StrD1_putamen,post=SNr_putamen,target='inh',synapse=DAPreCovariance_inhibitory_trace)
StrD1SNr_putamen.connect_from_matrix(weights_snr)
StrD1SNr_putamen.tau = 850 
StrD1SNr_putamen.regularization_threshold = 1.5
StrD1SNr_putamen.tau_alpha = 20.0
StrD1SNr_putamen.baseline_dopa = 2*baseline_dopa
StrD1SNr_putamen.K_dip = 0.01
StrD1SNr_putamen.K_burst = 1.0
StrD1SNr_putamen.threshold_post = 0.0
StrD1SNr_putamen.threshold_pre = 0.1
StrD1SNr_putamen.DA_type=1



#Indirect pathway
ObjStrD2_putamen = Projection(pre=Objectives[1:3],post=StrD2_putamen,target='exc',synapse=DAPostCovarianceNoThreshold)
ObjStrD2_putamen.connect_all_to_all(weights = Uniform(0.3,0.4)) 
ObjStrD2_putamen.tau = 60.
ObjStrD2_putamen.regularization_threshold = 1.0
ObjStrD2_putamen.tau_alpha = 1.0
ObjStrD2_putamen.baseline_dopa = 2*baseline_dopa
ObjStrD2_putamen.K_dip = 0.4
ObjStrD2_putamen.K_burst = 1.0
ObjStrD2_putamen.DA_type = -1

'''
ContextStrD2_putamen = Projection(pre=Context,post=StrD2_putamen,target='exc',synapse=DAPostCovarianceNoThreshold)
ContextStrD2_putamen.connect_all_to_all(weights = Uniform(0.3,0.4)) 
ContextStrD2_putamen.tau = 30.
ContextStrD2_putamen.regularization_threshold = 1.0
ContextStrD2_putamen.tau_alpha = 1.0
ContextStrD2_putamen.baseline_dopa = 2*baseline_dopa
ContextStrD2_putamen.K_dip = 0.4
ContextStrD2_putamen.K_burst = 1.0
ContextStrD2_putamen.DA_type = -1
'''
StrD2GPe_putamen = Projection(pre=StrD2_putamen,post=GPe_putamen,target='inh',synapse=DAPreCovariance_inhibitory)
StrD2GPe_putamen.connect_all_to_all(weights=Uniform(0.0,0.0001)) 
StrD2GPe_putamen.tau = 300.0
StrD2GPe_putamen.regularization_threshold = 2.0
StrD2GPe_putamen.tau_alpha = 1.0
StrD2GPe_putamen.baseline_dopa = 2*baseline_dopa
StrD2GPe_putamen.K_dip = 0.1
StrD2GPe_putamen.K_burst = 1.2
StrD2GPe_putamen.threshold_post = 0.05 
StrD2GPe_putamen.DA_type = -1

GPeSNr_putamen = Projection(pre=GPe_putamen,post=SNr_putamen,target='inh')
GPeSNr_putamen.connect_one_to_one(weights=0.1)


#Hyperdirect pathway
ObjSTN_putamen = Projection(pre=Objectives[1:3], post=STN_putamen, target='exc',synapse=DAPostCovarianceNoThreshold)
ObjSTN_putamen.connect_all_to_all(weights = Uniform(0.2,0.3)) 
ObjSTN_putamen.tau = 1000.0
ObjSTN_putamen.regularization_threshold = 0.4
ObjSTN_putamen.tau_alpha = 1.0
ObjSTN_putamen.baseline_dopa = 2*baseline_dopa
ObjSTN_putamen.K_dip = 0.4
ObjSTN_putamen.K_burst = 0.6
ObjSTN_putamen.DA_type = 1
ObjSTN_putamen.threshold_pre = 0.15

'''
ContextSTN_putamen = Projection(pre=Context, post=STN_putamen, target='exc',synapse=DAPostCovarianceNoThreshold)
ContextSTN_putamen.connect_all_to_all(weights = Uniform(0.2,0.3)) 
ContextSTN_putamen.tau = 600.0
ContextSTN_putamen.regularization_threshold = 0.4
ContextSTN_putamen.tau_alpha = 1.0
ContextSTN_putamen.baseline_dopa = 2*baseline_dopa
ContextSTN_putamen.K_dip = 0.4
ContextSTN_putamen.K_burst = 0.6
ContextSTN_putamen.DA_type = 1
ContextSTN_putamen.threshold_pre = 0.15
'''

STNSNr_putamen = Projection(pre=STN_putamen,post=SNr_putamen,target='exc',synapse=DAPreCovariance_excitatory)
STNSNr_putamen.connect_all_to_all(weights=Uniform(0.2,0.225)) 
STNSNr_putamen.tau = 1000.0
STNSNr_putamen.regularization_threshold = 1.3
STNSNr_putamen.tau_alpha = 1.0
STNSNr_putamen.baseline_dopa = 2*baseline_dopa
STNSNr_putamen.K_dip = 0.4
STNSNr_putamen.K_burst = 0.8 
STNSNr_putamen.thresholdpost = 0.15
STNSNr_putamen.DA_type = 1


#Local inhibition
StrD1StrD1_putamen = Projection(pre=StrD1_putamen,post=StrD1_putamen,target='inh')
StrD1StrD1_putamen.connect_all_to_all(weights = 1.0) #1.0

STNSTN_putamen = Projection(pre=STN_putamen,post=STN_putamen,target='inh')
STNSTN_putamen.connect_all_to_all(weights = 0.3)

StrD2StrD2_putamen = Projection(pre=StrD2_putamen,post=StrD2_putamen,target='inh')
StrD2StrD2_putamen.connect_all_to_all(weights=0.3)

StrThalStrThal_putamen = Projection(pre=StrThal_putamen,post=StrThal_putamen,target='inh')
StrThalStrThal_putamen.connect_all_to_all(weights=0.9)

SNrSNr_putamen = Projection(pre=SNr_putamen,post=SNr_putamen,target='exc',synapse=ReversedSynapse)
SNrSNr_putamen.connect_all_to_all(weights=0.6) #0.2

VAVA_putamen = Projection(pre=VA_putamen,post=VA_putamen,target='inh')
VAVA_putamen.connect_all_to_all(weights=0.9)

#Feedback from the thalamus
StrThalGPe_putamen = Projection(pre=StrThal_putamen,post=GPe_putamen,target='inh')
StrThalGPe_putamen.connect_one_to_one(weights=0.15) 

StrThalSNr_putamen = Projection(pre=StrThal_putamen,post=SNr_putamen,target='inh')
StrThalSNr_putamen.connect_one_to_one(weights=0.1)

VAStrThal_putamen = Projection(pre=PM,post=StrThal_putamen,target='exc')
VAStrThal_putamen.connect_one_to_one(weights=1.0)

#Output connections
SNrVA_putamen = Projection(pre=SNr_putamen,post=VA_putamen,target='inh')
SNrVA_putamen.connect_one_to_one(weights=1.0) 

VAPM_putamen = Projection(pre=VA_putamen,post=PM,target='exc')
VAPM_putamen.connect_one_to_one(weights=0.8)





#Reward system

SNcStrD1_put = Projection(pre=SNc_put,post=StrD1_putamen,target='dopa')
SNcStrD1_put.connect_all_to_all(weights=1.0)

SNcStrD2_put = Projection(pre=SNc_put,post=StrD2_putamen,target='dopa')
SNcStrD2_put.connect_all_to_all(weights=1.0)

SNcSNr_put = Projection(pre=SNc_put,post=SNr_putamen,target='dopa')
SNcSNr_put.connect_all_to_all(weights=1.0)

SNcSTN_put = Projection(pre=SNc_put,post=STN_putamen,target='dopa')
SNcSTN_put.connect_all_to_all(weights=1.0)

SNcGPe_put = Projection(pre=SNc_put,post=GPe_putamen,target='dopa')
SNcGPe_put.connect_all_to_all(weights=1.0)



PropSNc = Projection(pre=Propio[0:2],post=SNc_put,target='exc')
PropSNc.connect_one_to_one(weights=2.0) 


#Inhibition from the striatum to the dopaminergic cells
StrD1SNc_put = Projection(pre=StrD1_putamen,post=SNc_put,target='inh',synapse=DAPrediction)
StrD1SNc_put.connect_all_to_all(weights=0.0)
StrD1SNc_put.tau = 12000 
StrD1SNc_put.baseline_dopa = 0.1

SNcStrD1_caud0 = Projection(pre=SNc_caud[0],post=StrD1_caudate0,target='dopa')
SNcStrD1_caud0.connect_all_to_all(weights=1.0)
SNcStrD1_caud1 = Projection(pre=SNc_caud[1],post=StrD1_caudate1,target='dopa')
SNcStrD1_caud1.connect_all_to_all(weights=1.0)

SNcStrD2_caud0 = Projection(pre=SNc_caud[0],post=StrD2_caudate0,target='dopa')
SNcStrD2_caud0.connect_all_to_all(weights=1.0)
SNcStrD2_caud1 = Projection(pre=SNc_caud[1],post=StrD2_caudate1,target='dopa')
SNcStrD2_caud1.connect_all_to_all(weights=1.0)


SNcSNr_caud = Projection(pre=SNc_caud,post=SNr_caudate,target='dopa')
SNcSNr_caud.connect_all_to_all(weights=1.0)

SNcSTN_caud0 = Projection(pre=SNc_caud[0],post=STN_caudate0,target='dopa')
SNcSTN_caud0.connect_all_to_all(weights=1.0)
SNcSTN_caud1 = Projection(pre=SNc_caud[1],post=STN_caudate1,target='dopa')
SNcSTN_caud1.connect_all_to_all(weights=1.0)

SNcGPe_caud = Projection(pre=SNc_caud,post=GPe_caudate,target='dopa')
SNcGPe_caud.connect_all_to_all(weights=1.0)

PPTNSNc = Projection(pre=PPTN,post=SNc_caud,target='exc')
PPTNSNc.connect_one_to_one(weights=1.0)

StrD1SNc_caud0 = Projection(pre=StrD1_caudate0,post=SNc_caud[0],target='inh',synapse=DAPrediction)
StrD1SNc_caud0.connect_all_to_all(weights=0.0)
StrD1SNc_caud0.tau = 3000 

StrD1SNc_caud1 = Projection(pre=StrD1_caudate1,post=SNc_caud[1],target='inh',synapse=DAPrediction)
StrD1SNc_caud1.connect_all_to_all(weights=0.0)
StrD1SNc_caud1.tau = 3000 

#Cortical connections

PropioObj = Projection(pre=Propio[0:2],post=Objectives[1:3],target='exc')
PropioObj.connect_one_to_one(weights=1.2)

ObjObj = Projection(pre=Objectives[1:3],post=Objectives[1:3],target='inh')
ObjObj.connect_all_to_all(weights=0.5)

PMPM = Projection(pre=PM,post=PM,target='inh')
PMPM.connect_all_to_all(weights = 1.0)


#Hippo
HippoVA = Projection(pre=Hippo,post=VA_putamen[1],target='exc')
HippoVA.connect_all_to_all(weights = 1.0)

#Shortcut
ILVA = Projection(pre=IL,post=VA_putamen,target='exc')
ILVA.connect_one_to_one(weights=1.0)

VAIL = Projection(pre=VA_putamen,post=IL,target='exc')#,synapse=PreCovariance)
VAIL.connect_one_to_one(weights=0.05)


CorticoIL = Projection(pre=Input_neurons[0:10],post=IL,target='exc',synapse=PreCovariance)
CorticoIL.connect_all_to_all( weights = 0.6) 
CorticoIL.tau = 9000
CorticoIL.regularization_threshold = 3.0
CorticoIL.threshold_pre = 0.0
CorticoIL.threshold_post = 0.0



######################################################################################################################################################
######################################################################################################################################################



compile()



'''
Context - Action - Result mapping


Context 0:

action 0 - objective 1
action 1 - objective 2
action 2 - objective 3

Context 1:

action 0 - objective 3
action 1 - objective 1
action 2 - objective 2

Context 2:

action 0 - objective 2
action 1 - objective 3
action 2 - objective 1
'''

def achieved_objective(con,act):
    if(con==0):
        if act == 0:
            return 0
        if act == 1:
            return 1
    if(con==1):
        if act == 0:
            return 1
        if act == 1:
            return 0
    return -1



num_trials = 400

#arrays to store results
correct = np.zeros((2,int(num_trials/2))) #1 if a trial is correct, 0 if not correct. 2 types of trial, one per reward.
actions = np.zeros((2,int(num_trials/2))) #action taken on every trial
stim = 0
trial_stim = [0,0]




num_trials_per_stimuli = np.zeros(2)


last_trials = np.zeros((2,10))


saturation=False
saturation_trial = 1000000

complete = False
complete_trial = 0

overtraining = 80

for trial in range(num_trials):

    #if(trial>(num_trials-1)): #remove comment to lesion the striatum on the final trial
    #    StrD1_caudate.lesion = 0.0


    current_context = 0    
    #if(trial>(num_trials-1)):  #remove comment to have multiple contexts. 
    #    current_context = 1


    #switch the goal/stimulus every trial
    if stim==0:
        stim=1
    else:
        stim=0


    #input neuron initialization.
    Input_neurons.baseline = 0.0
    Context.baseline = 0.0
    Input_neurons.r = 0.0
    Objectives.r = 0.0
    Objectives.baseline = 0.0
    Objectives_extra.baseline=0.0
    Propio.baseline = 0.0
    SNc_caud.baseline = baseline_dopa

    #simulate inter trial period to let the network reach a stable state.
    simulate(700)

    #activate corresponding input neurons
    Context[current_context].baseline = 0.4
    Input_neurons[stim*2:(stim+1)*2].baseline = 0.5
    Input_neurons[15:18].baseline = 0.5
    Objectives_extra[stim*5:(stim+1)*5].baseline=1.0


    '''
    if(stim==1 and num_trials_per_stimuli[stim]>(saturation_trial+overtraining)): #remove comment to add devaluation trials
        Objectives_extra.baseline=0
        #Saturation.baseline = 1.0
    '''

    #initialize the state in the door
    Propio.r = 0
    Propio[2].baseline = 1.0

    #if(trial>(num_trials-1)): #remove comment to add the hippothalamus signal
    #    Hippo.baseline = 2.0
    #    #StrD1_putamen.lesion = 0.0

    #run to let the network reach a decision
    simulate(600)

    #read the response // soft max rule
    response = PM.r
    softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
    r = np.random.random()

    action = -1
    sum_probs = 0
    for i in range(2):
        sum_probs += softmax[i]            
        if r< sum_probs and action<0:
            action = i

        

    #Change the propio according to the action taken
    Propio[2].baseline=0.0
    ao = achieved_objective(current_context,action)
    Propio[ao].baseline=1.0

    #store the selected action
    actions[stim,trial_stim[stim]] = action

    #simulate to incorporate the environmental information
    simulate(40)

    #check if the trial is correct
    if(ao==stim):
        PPTN[stim].baseline = 0.5
        correct[stim,trial_stim[stim]] = 1
        last_trials[stim,int(num_trials_per_stimuli[stim])%10] = 1
    else:
        PPTN.baseline = 0.0
        last_trials[stim,int(num_trials_per_stimuli[stim])%10] = 0
    
    SNc_caud.firing = 1
    SNc_put.firing = 1

    #if on the devaluation trials no dopamine is introduced
    if(stim==1 and num_trials_per_stimuli[stim]>(saturation_trial+overtraining)): #if(num_trials_per_stimuli[stim]>trial_saturation and stim==1):
        SNc_caud.firing = 0
        SNc_put.firing = 0
        SNc_caud.baseline = 0



    simulate(100)


    PPTN.baseline = 0.0
    SNc_caud.firing = 0
    SNc_put.firing = 0


    num_trials_per_stimuli[stim]+=1

    #check if the 90%performance has been reached
    if(np.sum(last_trials[1])>=9 and saturation==False):
        saturation=True
        saturation_trial = trial_stim[1]

    if(np.sum(last_trials[0])>=9 and complete==False):
        complete = True
        complete_trial = trial_stim[0]

    trial_stim[stim]+=1

 
    

print('Finish simulation, now saving results')





#save results
np.save('correct_'+str(exp)+'.npy',correct)
np.save('actions_'+str(exp)+'.npy',actions)

np.save('num_correct_'+str(exp)+'.npy',[np.sum(correct[0,complete_trial+overtraining+1:complete_trial+overtraining+11]),np.sum(correct[1,saturation_trial+overtraining+1:saturation_trial+overtraining+11]) ])
np.save('s_trials_'+str(exp)+'.npy',[saturation_trial,complete_trial])

temp = saturation_trial+overtraining
np.save('num_correct_after_'+str(exp)+'.npy',[ np.sum(correct[1,temp+11:temp+21]),np.sum(correct[1,temp+21:temp+31]),np.sum(correct[1,temp+31:temp+41]),np.sum(correct[1,temp+41:temp+51]),np.sum(correct[1,temp+51:temp+61])])


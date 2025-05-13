#!usr/bin/env python3
# This Python script can be run with the following command on Lassen:
# lrun -T $threads_per_node python3 MakeInput.py $base_json
# It will abort if nnodes*threads_per_node /= numTiles
# Author: Tony Zahtila
# Modified to remove laser refinement by: Murray Cutforth


import argparse
import sys
import os
import json
import numpy as np
import h5py
from random import *
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.special import erf

# Parallelize
from joblib import Parallel, delayed
#from mpi4py import MPI
#mpiProcs = MPI.COMM_WORLD.Get_size()
#rank = MPI.COMM_WORLD.Get_rank()

import time

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen_new
import HTRrestart_new

from grid_generation import generate_grid, save_grid

parser = argparse.ArgumentParser()
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
parser.add_argument('--prefix', required=False, default='fluid_iter0000000000', type=str, help='location of IC files')
parser.add_argument('--only_json', required=False, default=0, type=int, help='Only write json file, not hdf files')
args = parser.parse_args()

##############################################################################
#                                 Setup Case                                 #
##############################################################################
# Misc stuff
figwidth = 9.0
fac_fig = 1.5
cm = 1/2.54  # centimeters in inches
#print(plt.rcParams.keys())  # Get all rcParams to aletr figure aesthetics
plt.rcParams['font.size'] = (10.0*fac_fig)
plt.rcParams['figure.figsize'] = (10.0*cm*fac_fig, 7.0*cm*fac_fig)
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['lines.linewidth'] = 1.0 * fac_fig
plt.rcParams['axes.linewidth'] = 1.0 * fac_fig
plt.rcParams['axes.labelsize'] = round(12 * fac_fig)
plt.rcParams['axes.titlesize'] = round(12 * fac_fig)
plt.rcParams['xtick.labelsize'] = round(10 * fac_fig)
plt.rcParams['ytick.labelsize'] = round(10 * fac_fig)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 3 * fac_fig
plt.rcParams['xtick.major.width'] = 0.8 * fac_fig
plt.rcParams['xtick.minor.size'] = 1.8 * fac_fig
plt.rcParams['xtick.minor.width'] = 0.6 * fac_fig
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 3 * fac_fig
plt.rcParams['ytick.major.width'] = 0.8 * fac_fig
plt.rcParams['ytick.minor.size'] = 1.8 * fac_fig
plt.rcParams['ytick.minor.width'] = 0.6 * fac_fig
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"text.usetex": True,"font.family": "Helvetica"})
i_fig = 0

tic0 = time.time()
write_probes = True

# Read base config
config = json.load(args.base_json)
prefix = args.prefix
only_json = args.only_json

# Input parameters
TInf  = config["Case"]["TInf"]
PInf  = config["Case"]["PInf"]
Tw    = config["Case"]["Tw"]
Lb    = config["Case"]["LengthBurner"]
Ln    = config["Case"]["LengthNozzle"]
Rb    = config["Case"]["RadiusBurner"]
Rn    = config["Case"]["RadiusNozzle"]
R_Ox    = config["Case"]["Radius_Ox"] # Radius of oxidizer core
R_F_in  = config["Case"]["Radius_F_in"] # Inner radius of fuel co-flow -- not used for NSCBC_Inflow
R_F_out = config["Case"]["Radius_F_out"] # Outer radius of fuel co-flow
mdot_Ox = config["Case"]["mDot_Ox"]
mdot_F = config["Case"]["mDot_F"]
PInj = config["Case"]["PInj"] # Static pressure of injected gas; only used by solver if flow is supersonic
if (("T_Ox" in config["Case"].keys()) and ("T_F" in config["Case"].keys())):
   T_Ox = config["Case"]["T_Ox"]
   T_F = config["Case"]["T_F"]
elif ("TInj" in config["Case"].keys()):
   T_Ox = config["Case"]["TInj"]
   T_F = config["Case"]["TInj"]
if ("XCH4_Inj" in config["Case"].keys()):
   XCH4_Inj = config["Case"]["XCH4_Inj"] # CH4 mole fraction in fuel stream
else:
   XCH4_Inj = 1.0
xBCLeft = config["BC"]["xBCLeft"]["type"]
if (xBCLeft == 'PSAAP_Inflow' or xBCLeft == 'PSAAP_RampedInflow'):
   U_bg = 0.0
else:
   U_bg = config["Case"]["U_background"] # Background inlet velocity, i.e. a "fake" co-flow for numerical stab.
thick = 1e-12 # config["Case"]["InjectorTransitionLength"] # Xk and T profile transition (not u) -- OVERRIDE.
nozzleTransitionRate = config["Case"]["nozzleTransitionRate"] # Nondimensional
xStretch = config["Case"]["xStretch"] # Stretch factor in x direction (tanh mesh)
xStretch_sigmoid1 = config["Case"]["xStretch_sigmoid1"] # 
xStretch_sigmoid2 = config["Case"]["xStretch_sigmoid2"] # 
radial_stretch_type = config["Case"]["radial_stretch_type"] # Stretch in y and z directions (either cosh or tanh)
yzStretch = config["Case"]["yzStretch"] # Stretch factor in y and z directions

# A_laser = config["Case"]["A_laser"] # Amplitude factor to govern laser strecth
# B_laser = config["Case"]["B_laser"] # Location factor to govern laser strecth
# C_laser = config["Case"]["C_laser"] # Spatial extent factor to govern laser strecth
# R_laser = config["Case"]["R_laser"]

if (radial_stretch_type == 'tanh'):
   yzStretch_position = config["Case"]["yzStretch_position"] # Location of transition from small dy (dz) to large dy (dz)
   yzStretch_ratio = config["Case"]["yzStretch_ratio"]       # Ratio between large dy (dz) and small dy (dz)
else:
   yzStretch_position = 0.0
   yzStretch_ratio = 1.0
if ("initWithFuel" in config["Case"].keys()):
   initWithFuel = config["Case"]["initWithFuel"]
else:
   initWithFuel = False
if ("nozzleStretch" in config["Case"].keys()):
   nozzleStretch = config["Case"]["nozzleStretch"]
else:
   nozzleStretch = 1.0
if ("squircleStretch" in config["Case"].keys()):
   squircleStretch = config['Case']['squircleStretch']
else:
   squircleStretch = 0.0
mixtureName = config['Flow']['mixture']['type']

# Derived parameters
L = Ln + Lb
Ar = Rb**2 / Rn**2 # Area ratio

# Safety...
if (XCH4_Inj == 1.0):
   XCH4_Inj -= 1e-12

# Announce
#if (rank == 0):
print('\n====================================================================================================')
print('\n                          ConvergingDuct initialization')
print('Here are the inputs you gave me:')
for arg in vars(args):
    print(arg,'=',getattr(args, arg))
print('')
print('mixtureName={}, initWithFuel={}'\
   .format(mixtureName,initWithFuel))
print('PInj={:9.3e}, T_Ox={:9.3e}, T_F={:9.3e}, XCH4_Inj={:9.3e}, mdot_Ox={:9.3e}, mdot_F={:9.3e}'\
   .format(PInj,T_Ox,T_F,XCH4_Inj,mdot_Ox,mdot_F))
print('xStretch={:9.3e}, yzStretch={:9.3e}, nozzleStretch={:9.3e}, squircleStretch={}, nozzleTransitionRate={}'\
   .format(xStretch,yzStretch,nozzleStretch,squircleStretch,nozzleTransitionRate))

# Mixture
if (mixtureName == 'ConstPropMix'):
   speciesNames = ['MIX']
elif (mixtureName == 'CH4O2InertMix'):
   speciesNames = ['CH4','O2']
elif (mixtureName == 'CH41StMix' or mixtureName == 'CH41StExtMix'):
   speciesNames = ['CH4','O2','CO2','H2O']
elif (mixtureName == 'BFERoxy6SpMix'):
   speciesNames = ['O2','H2O','CH4','CO','CO2','O']
elif (mixtureName == 'O2OMix'):
   speciesNames = ['O2','O']
elif (mixtureName == 'FFCM1_12SpMix' or mixtureName == 'FFCM1_12SpExtMix'):
   speciesNames = ['H2','H','O2','O','OH','HO2','H2O','CH3','CH4','CO','CO2','CH2O']
else:
   raise Exception('Unrecognized mixture name: {}'.format(mixtureName))
nSpec = len(speciesNames)
if (nSpec == 1):
   iCH4 = -1 # Should not be used
   iO2 = -1 # Should not be used
else:
   iCH4 = speciesNames.index('CH4')
   iO2 = speciesNames.index('O2')

# Gas properties
gamma_Ox = 1.4
gamma_F  = 1.32
mu_Ox  = 2.0567e-5 # Pa*s, from NIST.  (before it was 1.95e-5?)
mu_F   = 1.1248e-5 # Pa*s, from NIST.  (before it was 1.88e-5?)
Runiv = 8.3144598
W_Ox = 2*15.9994e-3
if (mixtureName == 'ConstPropMix'):
   W_F = W_Ox # Or else EOS relating rho_F, T_F will be wrong
else:
   W_F  = 4*1.00784e-3 + 12.0107e-3
p_Ox = PInj # Think of this as target static pressure, which should match chamber pressure if subsonic.
p_F = PInj # Think of this as target static pressure, which should match chamber pressure if subsonic.
rho_Ox = p_Ox/(Runiv/W_Ox*T_Ox)
rho_F = p_F/(Runiv/W_F*T_F)
c_Ox  = np.sqrt(gamma_Ox*Runiv/W_Ox*T_Ox) # Only used to compute Ma_Ox, for user's reference
c_F   = np.sqrt(gamma_F *Runiv/W_F *T_F) # Only used to compute Ma_F, for user's reference
A_Ox = np.pi * R_Ox**2 # cross-sectional area of O2 jet
A_F = np.pi * (R_F_out**2-R_F_in**2) # cross-sectional area of CH4 co-flow
U_Ox = mdot_Ox / (rho_Ox * A_Ox)
U_F  = mdot_F / (rho_F * A_F)
Ma_F  = U_F/c_F
Ma_Ox = U_Ox/c_Ox
Re_inj = rho_Ox * U_Ox * (2.0*R_F_out) / mu_Ox # Based on full radius and oxidizer properties, which has higher rho and U

# Reference quantities -- use to nondimensionalize when writing hdf file
Pbc = 101325.0 # Nozzle always exits into ambient pressure
if (nSpec == 1):
   Rgas = config["Flow"]["mixture"]["gasConstant"] #8.3145 / W_Ox # Based on O2 properties
   LRef = 1.0
   TRef = 1.0
   PRef = 1.0
   rhoRef = 1.0
   eRef = 1.0
   uRef = 1.0
   tRef = 1.0
   rhoInf = PInf/(Rgas*TInf) # Use for prescribing density profile
else:
   LRef = R_F_out
   TRef = 300.0 # Always
   PRef = 101325.0 # Always
   rhoRef = PRef/(Runiv/W_Ox*TRef) # Corresponds to XiRef={Pure O2}
   XiRef = {"Species" : [{"Name" : "O2", "MolarFrac" : 1.0 }]}
   eRef = PRef/rhoRef
   uRef = np.sqrt(eRef)
   tRef = LRef/uRef
   rhoInf = PInf/(Runiv/W_Ox*TInf) # Corresponds to XiRef={Pure O2}

# Announce
#if (rank == 0):
print('')
print('nSpec={}, rhoRef={:9.3e} kg/m^3, uRef={:9.3e} m/s, eRef={:9.3e} J/kg, tRef={:9.3e} s, LRef={:9.3e} m'\
   .format(nSpec,rhoRef,uRef,eRef,tRef,LRef))
print('U_F={:9.3e} m/s,  Ma_F={:9.3e},  rho_F={:9.3e} kg/m^3,  A_F={:9.3e},  T_F={:9.3e},  c_F={:9.3e} m/s'\
   .format(U_F,Ma_F,rho_F,A_F,T_F,c_F))
print('U_Ox={:9.3e} m/s, Ma_Ox={:9.3e}, rho_Ox={:9.3e} kg/m^3, A_Ox={:9.3e}, T_Ox={:9.3e}, c_Ox={:9.3e} m/s,'\
   .format(U_Ox,Ma_Ox,rho_Ox,A_Ox,T_Ox,c_Ox))
print('Re_inj={:9.3e}, PInf={:9.3e} Pa, TInf={:9.3e} K, PInj={:9.3e}, mu_F={:9.3e}, mu_Ox={:9.3e}'\
   .format(Re_inj,PInf,TInf,PInj,mu_F,mu_Ox))
print('')

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"]["type"] == "Restart"
config["Flow"]["initCase"]["restartDir"] = prefix
config["Grid"]["GridInput"]["gridDir"] = prefix + '/grid'
#if (rank == 0):
if not os.path.exists(config["Flow"]["initCase"]["restartDir"]):
   os.makedirs(config["Flow"]["initCase"]["restartDir"])
if not os.path.exists(config["Grid"]["GridInput"]["gridDir"]):
   os.makedirs(config["Grid"]["GridInput"]["gridDir"])

# Set reference quantities
if (nSpec > 1):
   config["Flow"]["mixture"]["LRef"] = LRef #R_F_out
   config["Flow"]["mixture"]["PRef"] = PRef #PInf
   config["Flow"]["mixture"]["TRef"] = TRef #TInf
   config["Flow"]["mixture"]["XiRef"] = XiRef #{"Species" : [{"Name" : "O2", "MolarFrac" : 1.0 }]}
#config["Integrator"]["EulerScheme"]["vorticityScale"] = abs(U_Ox-U_F) / LRef #(U_Ox - U_F)/1.0

# Grid size
xNum = config["Grid"]["xNum"]
yNum = config["Grid"]["yNum"]
zNum = config["Grid"]["zNum"]

# Inlet displacement thickness
#h = Ri_F #mu_Ox*ReIn/((U_F-U_Ox)*rho_Ox)
#config["Case"]["ReInlet"] = Ri_F*(U_Ox-U_F)*rho_Ox/mu_Ox
# Rescale quantities
#U_F  *= np.sqrt(rho_Ox/PInf)
#U_Ox *= np.sqrt(rho_Ox/PInf)
config["Flow"]["mixture"]["LRef"] = LRef #Ri_F
config["Flow"]["mixture"]["PRef"] = PRef #PInf
config["Flow"]["mixture"]["TRef"] = TRef #TInf
config["Flow"]["mixture"]["XiRef"] = XiRef #{"Species" : [{"Name" : "O2", "MolarFrac" : 1.0 }]}
#config["Integrator"]["EulerScheme"]["vorticityScale"] = abs(U_Ox-U_F) / LRef #(U_Ox - U_F)/1.0

# Change mdot so that PSAAP_InflowBC will calculate inflow velocity in 2D matching that in 3D.
# In other words, the velocity corresponding to the input mdot in 3D is calculated, and then
# mdot is modified so that the same velocity should be recovered when run in 2D.
if (zNum == 1):
   mdot_Ox *= LRef * 2.0/(np.pi*R_Ox)
   mdot_F *= LRef * 2.0*(R_F_out-R_F_in)/(np.pi*(R_F_out**2-R_F_in**2))
   #if (rank == 0):
   print('Mass flows modified for 2D: mdot_Ox={:9.3e}, mdot_F={:9.3e}'.format(mdot_Ox,mdot_F))

# Check that partitioning is possible; use xNum, NOT xGrid.size
assert config['Grid']['xNum'] % config["Mapping"]["tiles"][0] == 0
assert config['Grid']['yNum'] % config["Mapping"]["tiles"][1] == 0
assert config['Grid']['zNum'] % config["Mapping"]["tiles"][2] == 0
NxTile = int(config['Grid']['xNum']/(config["Mapping"]["tiles"][0]/config['Mapping']['tilesPerRank'][0]))
NyTile = int(config['Grid']['yNum']/(config["Mapping"]["tiles"][1]/config['Mapping']['tilesPerRank'][1]))
NzTile = int(config['Grid']['zNum']/(config["Mapping"]["tiles"][2]/config['Mapping']['tilesPerRank'][2]))

# Load mapping
assert config['Mapping']['tiles'][0] % config['Mapping']['tilesPerRank'][0] == 0
assert config['Mapping']['tiles'][1] % config['Mapping']['tilesPerRank'][1] == 0
assert config['Mapping']['tiles'][2] % config['Mapping']['tilesPerRank'][2] == 0
numTiles = [ int(config["Mapping"]["tiles"][i]/config['Mapping']['tilesPerRank'][i]) for i in range(3) ]
NxTile = int(config["Grid"]["xNum"]/numTiles[0])
NyTile = int(config["Grid"]["yNum"]/numTiles[1])
NzTile = int(config["Grid"]["zNum"]/numTiles[2])

# Number of processors on a single node
nproc_max = 40 # 40 CPUs on Lassen
nproc = min(nproc_max,int(np.prod(numTiles)))
print('nproc=',nproc)
print('numTiles=',numTiles)
print('Points per tile = {}'.format(xNum*yNum*zNum / np.prod(numTiles)))

# Determine number of halo points
halo = [0, 0, 0]
if config["BC"]["xBCLeft"]["type"] == "Periodic":
   assert config["BC"]["xBCLeft"]["type"] == config["BC"]["xBCRight"]["type"]
else:
   halo[0] = 1
if config["BC"]["yBCLeft"]["type"] == "Periodic":
   assert config["BC"]["yBCLeft"]["type"] == config["BC"]["yBCRight"]["type"]
else:
   halo[1] = 1
if config["BC"]["zBCLeft"]["type"] == "Periodic":
   assert config["BC"]["zBCLeft"]["type"] == config["BC"]["zBCRight"]["type"]
else:
   halo[2] = 1

##############################################################################
#                            Boundary conditions                             #
##############################################################################
assert config["BC"]["xBCLeft"]["VelocityProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = prefix
assert config["BC"]["xBCLeft"]["TemperatureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = prefix
assert config["BC"]["xBCLeft"]["MixtureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["MixtureProfile"]["FileDir"] = prefix
config["BC"]["xBCLeft"]["P"] = PInj/PRef # only matters if supersonic
if (xBCLeft == "PSAAP_Inflow" or xBCLeft == "PSAAP_RampedInflow"):
   config["BC"]["xBCLeft"]["Radius1"] = R_Ox/LRef
   config["BC"]["xBCLeft"]["Radius2"] = R_F_in/LRef
   config["BC"]["xBCLeft"]["Radius3"] = R_F_out/LRef
   config["BC"]["xBCLeft"]["mDot1"] = mdot_Ox/(rhoRef*uRef*LRef**2)
   config["BC"]["xBCLeft"]["mDot2"] = mdot_F/(rhoRef*uRef*LRef**2)

assert config["BC"]["xBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["xBCRight"]["P"] = PInf/PRef

assert config["BC"]["yBCLeft"]["type"]  == "IsothermalWall"
assert config["BC"]["yBCLeft"]["TemperatureProfile"]["type"] == "Constant"
config["BC"]["yBCLeft"]["TemperatureProfile"]["temperature"] = Tw/TRef

assert config["BC"]["yBCRight"]["type"] == "IsothermalWall"
assert config["BC"]["yBCRight"]["TemperatureProfile"]["type"] == "Constant"
config["BC"]["yBCRight"]["TemperatureProfile"]["temperature"] = Tw/TRef

if (zNum > 1):
   assert config["BC"]["zBCLeft"]["type"]  == "IsothermalWall"
   assert config["BC"]["zBCLeft"]["TemperatureProfile"]["type"] == "Constant"
   config["BC"]["zBCLeft"]["TemperatureProfile"]["temperature"] = Tw/TRef

   assert config["BC"]["zBCRight"]["type"] == "IsothermalWall"
   assert config["BC"]["zBCRight"]["TemperatureProfile"]["type"] == "Constant"
   config["BC"]["zBCRight"]["TemperatureProfile"]["temperature"] = Tw/TRef

##############################################################################
# Output json files                                                          #
##############################################################################
if (zNum == 1):
   out_json = 'GG-combustor-2D.json'
else:
   out_json = 'GG-combustor.json'
#if (rank == 0):
with open(out_json, 'w') as fout:
   json.dump(config, fout, indent=3)
print('File written: {}\n'.format(out_json))

if (only_json == 1):
   print('Exiting now --- only json file requested.')
   exit()

##############################################################################
# Generate Grid                                                              #
##############################################################################

# I have refactored this section into the grid_generation.py functions
# I removed the laser location refinement, so that we use the same grid regardless of laser focus
# This is necessary for the DGP project as we aim the laser throughout entire combustor

tic = time.time()
nodes = generate_grid(config, show_plots=False)
save_grid(config, nodes)



##############################################################################
#                              Write probes
##############################################################################

# Find indices for probes
# Note -- beware of concave x-planes in nozzle region
if (write_probes):
   # Six wall probes from experiments
   # Because lines of constant y and z curve in the x direction, and because we want to be 100% sure that
   # boundary points are selected, we specify the locations xp and indices jp, kp and calculate the x-index ip
   jmid = int(round(yNum/2))
   kmid = int(round(zNum/2))
   xp = [9.5e-3, 9.5e-3, 9.5e-3, 9.5e-3, 39.9e-3, 70.2e-3]
   jp = [    0,  jmid,   yNum,   jmid,   jmid,   jmid]
   kp = [ kmid,     0,   kmid,   zNum,   zNum,   zNum]
   ip = [ 0 for i in range(len(xp)) ]
   for l in range(len(xp)):
      d = np.abs(nodes[kp[l],jp[l],:,0] - xp[l]/LRef)
      ind = np.unravel_index(np.argmin(d,axis=None),d.shape)
      ip[l] = int(ind[0])

   # Collect probe indices
   probes = []
   for l in range(len(xp)):
      fromCell = [ip[l],   jp[l],   kp[l]  ]
      uptoCell = [ip[l]+1, jp[l]+1, kp[l]+1]
      probes.append({ "fromCell" : [ int(fromCell[i]) for i in range(3) ], \
                      "uptoCell" : [ int(uptoCell[i]) for i in range(3) ] }) # must convert to int for json
      pnode = nodes[kp[l],jp[l],ip[l]]
      print('probe {}:  fromCell={};  uptoCell={};  [xp,yp,zp]=[{:.2f}, {:.2f}, {:.2f}]'
         .format(l,fromCell,uptoCell,pnode[0],pnode[1],pnode[2]))

   # Reread json and add probes
   with open(out_json, 'r') as fin:
      config = json.load(fin)
      config['IO']['probes'] = probes

   # Write
   with open(out_json, 'w') as fout:
      json.dump(config, fout, indent=3)
   print('File written with probes: {}\n'.format(out_json))

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
def tanhStep(r,r0,th):# Transitions from 0.1 to 0.9 in th, with asymptotic values [0,1] for r=[-inf,inf]
   c = np.arctanh(2*0.9 - 1)
   return 0.5*(np.tanh(2*c/th*(r-r0)) + 1)
def tanhBump(r,r0,d,th): # Bump at r=r0 with width d, which transitions from 0.1 to 0.9 in thickness th
   return tanhStep(r,r0-d/2,th) + tanhStep(r,r0+d/2,-th) - 1

# Used for interior specification
cCshape=(nodes.shape[0]+1,nodes.shape[1]+1,nodes.shape[2]+1)
centerCoord = np.ndarray(cCshape, dtype=np.dtype('(3,)f8'))
centerCoord[1:-1,1:-1,1:-1,:]= (nodes[0:-1,1:  ,1:  ] + nodes[1:  ,1:  ,1:  ] +
                                nodes[0:-1,0:-1,1:  ] + nodes[1:  ,0:-1,1:  ] +
                                nodes[0:-1,1:  ,0:-1] + nodes[1:  ,1:  ,0:-1] +
                                nodes[0:-1,0:-1,0:-1] + nodes[1:  ,0:-1,0:-1])/8
centerCoord[:,:,0 ,:]=2*centerCoord[:,:,1 ,:]-centerCoord[:,:, 2,:]
centerCoord[:,:,-1,:]=2*centerCoord[:,:,-2,:]-centerCoord[:,:,-3,:]
centerCoord[:,0 ,:,:]=  centerCoord[:,1 ,:,:]
centerCoord[:,-1,:,:]=  centerCoord[:,-2,:,:]
centerCoord[0 ,:,:,:]=  centerCoord[0 ,:,:,:]
centerCoord[-1,:,:,:]=  centerCoord[-1,:,:,:]

# Profiles
if (xBCLeft == 'PSAAP_Inflow' or xBCLeft == "PSAAP_RampedInflow"):

   # Velocity profile
   def velocityProfile_r(y,U0,U1,U2,w): # U0 not used, but keep same arg list as NSCBC_Inflow to avoid if statement later
      R1 = R_Ox/LRef
      R2 = R_F_in/LRef
      R3 = R_F_out/LRef
      cond1 =  y<=R1
      cond2 = (y>=R2) & (y<=R3)
      u = np.zeros(y.shape)
      u[cond1]=-U1*np.tanh((y[cond1]-R1)/w)
      u[cond2]=-U2*np.tanh((y[cond2]-R3)/w)*np.tanh((y[cond2]-R2)/w)
      return u


   # Smoothed scalar profile
   def scalarProfile_r(r,F0,F1,F2,w,offset):
      R1 = R_Ox/LRef
      R2 = R_F_in/LRef
      R3 = R_F_out/LRef
      return F0 + (F1-F0)*tanhBump(r,0,2*(R1-offset),w) + (F2-F0)*tanhBump(r,(R2+R3)/2,R3-R2-2*offset,w)

elif (xBCLeft == 'NSCBC_Inflow'): # This has same arguments as PSAAP_Inflow profile, to avoid if statement later

   # Velocity profile at inlet
   def velocityProfile_r(r,U0,U1,U2,w):
      R1 = R_Ox/LRef
      R2 = R_F_in/LRef
      R3 = R_F_out/LRef
      return (U1-U0)*tanhBump(r,0,2*R1,w) + (U2-U0)*tanhBump(r,(R2+R3)/2.0,R3-R2,w) + U0*tanhBump(r,0,2*2*R3,w)

   # For temperature profile
   def scalarProfile_r(r,F0,F1,F2,w,offset):
      R1 = R_Ox/LRef
      R2 = R_F_in/LRef
      R3 = R_F_out/LRef
      return F0 + (F1-F0)*tanhBump(r,0,2*(R1-offset),w) + (F2-F0)*tanhBump(r,(R2+R3)/2,R3-R2-2*offset,w)


def u_p_t_Xi(lo,hi,shape):
   u = np.zeros(shape, dtype = np.dtype("(3,)f8"))
   p = np.zeros(shape)
   T = np.zeros(shape)
   Xi= np.zeros(shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
   u[:] =[0.0, 0.0, 0.0]
   p[:] = PInf/PRef
   T[:] = TInf/TRef
   Xi[:] = 1e-60
   Xi[:,:,:,iO2]= 1.0
   return u,p,T,Xi

def T_Xi_u_prof(lo,hi,shape):
   T = np.zeros(shape)
   T += TInf/TRef
   Xi= np.zeros(shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
   u = np.zeros(shape, dtype = np.dtype("(3,)f8"))
   cC=centerCoord[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0],:]
   r = np.sqrt(cC[:,:,:,1]**2 + cC[:,:,:,2]**2)
   Xi[:,:,:,iCH4]=scalarProfile_r(r, 1e-60, 1e-60, XCH4_Inj, thick, thick/2.0)
   Xi[:,:,:,iO2] = 1.0 - Xi[:,:,:,iCH4]
   u[:,:,:,0]=velocityProfile_r(r, U_bg/uRef, U_Ox/uRef, U_F/uRef, 0.03)
   return T,Xi,u

def AvgNumD(lo,hi,shape):
   return 0.0

def AvgNumI(lo,hi,shape):
   return 0.0

def AvgDenD(lo,hi,shape):
   return 0.0

def AvgDenI(lo,hi,shape):
   return 0.0

#def fturbVx(lo,hi,shape):
#   return 0.0

def fturbVx(lo_bound,hi_bound,shape):
   turbVx = np.zeros(shape, dtype = np.dtype("(3,)f8")) 
   return turbVx
#   return turbVx[lo_bound[2]:hi_bound[2],lo_bound[1]:hi_bound[1],lo_bound[0]:hi_bound[0],:]
#   return np.zeros(shape, dtype = np.dtype("(3,)f8"))

print('Expensive write bit started')
# Write the files
tic = time.time()
restart = HTRrestart_new.HTRrestart(config)
#restart.write_fast(prefix, nSpec,
#                   u_p_t_Xi,AvgNumD, AvgNumI,
#                   AvgDenD, AvgDenI,
#                   T_Xi_u_prof=T_Xi_u_prof,
#                   nproc = 4)

restart.write_fast(prefix, nSpec=nSpec,
                  AvgNumD=AvgNumD, AvgNumI=AvgNumI,
                  AvgDenD=AvgDenD, AvgDenI=AvgDenI,
                  turbVx=fturbVx,
                  u_p_t_Xi = u_p_t_Xi,
                  T_Xi_u_prof=T_Xi_u_prof,
                  nproc = 4)

print('Initial condition written to {}  ({:.1f} s)'.format(config["Flow"]["initCase"]["restartDir"],time.time()-tic))

# All done.
print('Done.  Total time elapsed: {:.1f} s'.format(time.time()-tic0))

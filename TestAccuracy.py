from glob import iglob 
import os
from icecream import ic
from utils import LoadOnlyLandmarks
import torch
import numpy as np

def search(path,*args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments=[]
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {key: sorted([i for i in iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)]) for key in arguments}

def GetPatients(test_files,ref_files):
    """To associate scan and json files to every patient in input folder of SEMI ASO"""

    patients = {}

    for i in range(len(test_files)):
        patients = AddInDict(test_files, patients, i, 'test')
        patients = AddInDict(ref_files, patients, i, 'ref')
        
    return patients

def AddInDict(json_files, patients, i,key):
    patientjson = os.path.basename(json_files[i]).split('_Or')[0].split('_OR')[0].split('_lm')[0].split('_Scanreg')[0].split('.')[0]

    if patientjson not in patients.keys():
        patients[patientjson] = {key:json_files[i]}
    else:
        patients[patientjson][key] = json_files[i]
    
    return patients


def MSELoss(test_dir,ref_dir):

    MSE = torch.nn.MSELoss() # Loss

    test_json = search(test_dir,'json')['json']
    ref_json = search(ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)

    ldmk_list = ['LOr','ROr','LPo','N','RPo','S']
    
    TEST = {key:[] for key in ldmk_list}
    REF = {key:[] for key in ldmk_list}

    for patient,data in patients.items():

        if "test" in data.keys() and "ref" in data.keys():

            test = LoadOnlyLandmarks(data["test"],ldmk_list=ldmk_list)
            ref = LoadOnlyLandmarks(data["ref"],ldmk_list=ldmk_list)

            for lm in ldmk_list:
                if lm in test.keys() and lm in ref.keys():
                    TEST[lm].append(test[lm])
                    REF[lm].append(ref[lm])
    
    for lm in ldmk_list:
        print("For landmark: {}".format(lm))
        print(MSE(torch.tensor(np.array(TEST[lm])),torch.tensor(np.array(REF[lm]))))

def normal_vector(data, plane_list):
    points = []
    for lm in plane_list:
        try:
            points.append(data[lm])
        except KeyError:
            continue
    X = np.array(points)
    
    U,S,V = np.linalg.svd(X)
    
    return V[-1]

def angles(v1,v2,type=None):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    
    angle = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    pitch = np.arcsin(cross[2] / (np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(angle)))

    roll = np.arcsin(cross[1] / (np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(angle)))

    yaw = np.arcsin(cross[0] / (np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(angle)))
    
    if type == 'pitch':print("pitch:",pitch)
    if type == 'roll':print("roll:",roll)
    if type == 'yaw':print("yaw:",yaw)
    else:
        print("pitch:",pitch)
        print("roll:",roll)
        print("yaw:",yaw)
def OrientationError():
    
    ldmk_list = ['LOr','ROr','LPo','N','RPo','S']
    
    plane1 = ['LOr','ROr','LPo','RPo']
    plane2 = ['Ba','N','S']

    PLANE1_TEST = []
    PLANE2_TEST = []
    PLANE1_REF = []
    PLANE2_REF = []
    NAME = []

    test_json = search(test_dir,'json')['json']
    ref_json = search(ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)


    for patient,data in patients.items():

        if "test" in data.keys() and "ref" in data.keys() :#and patient == 'MAMP_0002_T2':
            
            test = LoadOnlyLandmarks(data["test"],ldmk_list=ldmk_list)
            ref = LoadOnlyLandmarks(data["ref"],ldmk_list=ldmk_list)
            test1,test2,ref1,ref2 = normal_vector(test,plane1),normal_vector(test,plane2),normal_vector(ref,plane1),normal_vector(ref,plane2)
            # print("For patient: ",patient)
            # print("Plane 1:")
            # angles(test1,ref1)
            # print("Plane 2:")
            # angles(test2,ref2)
            # print("="*70)
            PLANE1_TEST.append(test1),PLANE2_TEST.append(test2),PLANE1_REF.append(ref1),PLANE2_REF.append(ref2),NAME.append(patient)

    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: 1 - abs(CosSim(torch.Tensor(x),torch.Tensor(y)))
    
    plane1Loss = np.array(Loss(np.array(PLANE1_TEST),np.array(PLANE1_REF)))
    plane2Loss = np.array(Loss(np.array(PLANE2_TEST),np.array(PLANE2_REF)))
    print("="*70)
    print("For plane 1: ",np.mean(plane1Loss))
    print("For plane 2: ",np.mean(plane2Loss))

if __name__ == '__main__':
    
    test_dir = '/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/ASO_Output/'
    ref_dir = '/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Felicia_Oriented/'

    # MSELoss(test_dir,ref_dir)

    OrientationError()
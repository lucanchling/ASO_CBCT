from glob import iglob 
import os
from utils import LoadOnlyLandmarks
import torch
import numpy as np
import pandas as pd
import argparse

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

def OrientationError(args,ldmk_list,plane1,plane2):
    PLANE1_TEST,PLANE2_TEST,PLANE1_REF,PLANE2_REF,NAME,NB_LDMK = [],[],[],[],[],[]

    test_json = search(args.test_dir,'json')['json']
    ref_json = search(args.ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)

    df = pd.DataFrame([patient for patient,data in patients.items() if  "test" in data.keys() and "ref" in data.keys()], columns=['Patients'])
    for patient,data in patients.items():

        if "test" in data.keys() and "ref" in data.keys() :
            
            test = LoadOnlyLandmarks(data["test"],ldmk_list=ldmk_list)
            ref = LoadOnlyLandmarks(data["ref"],ldmk_list=ldmk_list)
            test1,test2,ref1,ref2 = normal_vector(test,plane1),normal_vector(test,plane2),normal_vector(ref,plane1),normal_vector(ref,plane2)
            # print("For patient: ",patient)
            # print("Plane 1:")
            # angles(test1,ref1)
            # print("Plane 2:")
            # angles(test2,ref2)
            # print("="*70)
            PLANE1_TEST.append(test1),PLANE2_TEST.append(test2),PLANE1_REF.append(ref1),PLANE2_REF.append(ref2),NAME.append(patient),NB_LDMK.append(len(test))

    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: abs(CosSim(torch.Tensor(x),torch.Tensor(y)))
    plane1Loss = np.array(Loss(np.array(PLANE1_TEST),np.array(PLANE1_REF)))
    plane2Loss = np.array(Loss(np.array(PLANE2_TEST),np.array(PLANE2_REF)))
    
    degplane1 = np.arccos(plane1Loss)*180/np.pi
    degplane2 = np.arccos(plane2Loss)*180/np.pi
    
    #df['Plane1'] = plane1Loss
    df['Degre1'] = degplane1
    #df['Plane2'] = plane2Loss
    df['Degre2'] = degplane2
    df['Nb Landmark'] = NB_LDMK

    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)

    df.to_csv(args.csv_dir+args.csv_name.split('.')[0]+'.csv',index=False)    
    
    print("="*70)
    print("For plane 1: ",np.mean(plane1Loss))
    print("For plane 2: ",np.mean(plane2Loss))

def GetLandmarks(type_accuracy):
    if type_accuracy == 'head':
        ldmk_list = ['Ba','LOr','ROr','LPo','N','RPo','S']
        plane1 = ['LOr','ROr','LPo','RPo']
        plane2 = ['Ba','N','S']

    if type_accuracy == 'max':
        ldmk_list = ['ANS','PNS','IF','UR6O','UL6O','UR1O','UR6_UL6','UR1_UL1']
        plane1 = ['UR6O','UL6O','UR1O']
        plane2 = ['ANS','PNS','IF','UR6_UL6','UR1_UL1']

    if type_accuracy == 'mand':
        ldmk_list = ['LL6O','LR6O','LR1O','B','Pog','Me']
        plane1 = []
        plane2 = []

    return ldmk_list,plane1,plane2

    # if type_accuracy == 'maxillary':

    # if type_accuracy == 'mandible':

def main(args):
    print("Test directory: ",args.test_dir)

    ldmk_list,plane1,plane2 = GetLandmarks(args.type_acc)

    OrientationError(args,ldmk_list,plane1,plane2)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', type=str, default='/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Head/Felicia_BAMP_ASO_OUTPUT/', help='Path to the test directory')
    parser.add_argument('--ref_dir', type=str, default='/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Head/Felicia_BAMP_Oriented/', help='Path to the reference directory')
    parser.add_argument('--csv_name', type=str, default='Accuracy', help='Name of the csv file')
    parser.add_argument('--csv_dir', type=str, default='./', help='Path to the csv directory')
    parser.add_argument('--type_acc', type=str, help='Type of accuracy to compute', default='head')

    args = parser.parse_args()
    
    main(args)
    
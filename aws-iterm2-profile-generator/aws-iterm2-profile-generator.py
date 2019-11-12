#!/usr/local/bin/python
# coding: UTF-8
"""
Script to automatically generate an iTerm2 dynamic profile file and/or a SSH Config include file based from your EC2 resources.

python3 iterm2-update-profiles-ec2.py --iterm --sshconfig --keypath '~/Project/.ssh/'
where:
    --iterm instructs the script to generate an iterm2 dynamic profile of the detected resources,
    --sshconfig generates a SSH config file using basic details, additional bits beyond the bare minimum can be added down in the code below
    --keypath '~/relativepath/' is the path the keyfiles are stored and the location to save the ssh config if generated. You will need to add this as an include to your normal SSH config

Troubleshooting
1) This uses the aws cli to describe the EC2 instances, if your AWS_PROFILE and AWS_DEFAULT_REGION are not set, this isn't going to work
2) This assumes your EC2 resources have a Name, Environment and Application tag as the resultant menu structure is based on these.

"""


import boto3
import json
import argparse
from pathlib import Path
from os import path
from excludes import username_overrides, key_overrides, iterm2_default_profile, exclude_instances

def getEC2Instances():
    # Create a dictionary for the instances
    instances = {}
    instance_names = []

    # Create a client and filter for boto
    reg_client = boto3.client('ec2')
    reg_response = reg_client.describe_regions()
   
    for region in reg_response['Regions']:
        # print("Checking through the " + region['RegionName'])

        client = boto3.client('ec2',  region_name=region['RegionName'])
        response = client.describe_instances(
                Filters = [
                    {'Name':'instance-state-name', 'Values': ['running']}
                ]
        )

        print("Checking through the " + region['RegionName'] + " (" + str(len(response['Reservations'])) + ")")

        # for each insatnce in the reservation process the tags for the ones we expect.
        for reservation in response['Reservations']:
                for instance in reservation['Instances']:

                    # Find the instance ID
                    instance_id = instance['InstanceId']

                    if instance_id in exclude_instances:
                        print("\tSkipping instance" + instance_id)
                        break

                    # Loop throught the tags object and find the ones we are interested in, to add more tags do it here
                    for tag in instance['Tags']:
                            if tag['Key'] == 'Name':
                                    name = tag['Value']

                            if tag['Key'] == 'Application':
                                    application = tag['Value']
                                    
                            if tag['Key'] == 'Environment':
                                    environment = tag['Value']

                    # If there is an override for the key name, apply it
                    if name in key_overrides:
                        keyname = key_overrides.get(name)
                    else:
                        keyname = instance['KeyName']
                    
                    # If there is an override for the username, apply it
                    if name in username_overrides:
                        username = username_overrides.get(name)
                    else:
                        username = 'ec2-user'

                    if name in instance_names:
                        print("\t" + name + " looks to be a duplicate, renaming as " + name + '_' + instance_id) 
                        instance['name'] = name + "_" + instance_id


                    instance_names.append(name)

                    # Check to see if the key exists
                    if not path.exists(keypath + keyname):
                        print("\tMissing keyfile '" + keyname +"' the instance '" + name + "' seems to use it")


                    # Add the instance to the instanses object using the following attributes
                    instances[instance_id] = {
                        'name':name,
                        'environment':environment,
                        'application':application,
                        'keyname':keyname,
                        'username':username,
                        'region':region['RegionName'],
                        'privateip':instance['PrivateIpAddress'],
                        'privatednsname':instance['PrivateDnsName']
                    }

                    
    return instances


def update_iterm(instances):
    instance_names = []
    handle = open(str(Path.home()) + "/Library/Application Support/iTerm2/DynamicProfiles/aws", 'wt+')

    profiles = []

    for instance_i in instances:
        instance = instances[instance_i]
        
        if instance['name'] in instance_names:
            print(instance['name'] + " looks to be a duplicate, renaming as " + instance['name'] + '_' + instance_i) 
            instance['name'] = instance['name'] + "_" + instance_i


        instance_names.append(instance['name'])

        profile = {
                "Name":instance['name'],
                "Guid":instance['name'],
                "Badge Text":instance['environment'],
                "Tags":["Dynamic/" + instance['region'] + "/" + instance['environment'] + "/" + instance['application']],
                "Dynamic Profile Parent Name": iterm2_default_profile,
                "Custom Command" : "Yes",
                "Command" : "ssh -oStrictHostKeyChecking=no -oUpdateHostKeys=yes " + instance['username'] + "@" + instance['privateip'] + " -i " + keypath + "/" + instance['keyname'] 
                }

        profiles.append(profile)

    profiles = {"Profiles":profiles} 
    handle.write(json.dumps(profiles,sort_keys=True,indent=4, separators=(',', ': ')))
    handle.close()


def update_sshconfig(instances):
    

    handle = open(keypath + "config_aws", 'wt+')
    profiles = ""     

    for instance_i in instances:
        instance = instances[instance_i]

        profile = """Host "{name}" {privatedns}
    Hostname {ip}
    IdentityFile "{keyfile}"
    user {user}
    \n""".format(
            name=instance['name'],
            privatedns=instance['privatednsname'],
            ip=instance['privateip'],
            keyfile=keypath + instance['keyname'],
            user=instance['username']
        )

        # profiles.append(profile)
        profiles = profiles + "\n" + profile
    
    handle.write(profiles)
    handle.close()



#---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--keypath',  help='Path to folder where the SSH keys are, default is ~/.ssh/', default='~/.ssh/')
parser.add_argument('--iterm',  help='Generate iterm2 dynamic config', action='store_true')
parser.add_argument('--sshconfig', help='Generate SSH Config file', action='store_true')
args = parser.parse_args()

# Switch out any relative path usage for the home directory, the script uses open() later which doesnt play well with relative paths
keypath = (args.keypath).replace('~', str(Path.home()) )

# Store the instances using the current AWS_PROFILE details
print("Getting EC2 instance inventory...")

instances = getEC2Instances()


# If --iterm was passed, generate the iterm2 dynamic profile config
if args.iterm:
    print("Generating iTerm2 dynamic profiles file")
    
    update_iterm(instances)

# If --sshconfig was passed, generate an ssh config file
if args.sshconfig:
    print("Generating SSH Config dynamic config file in.", keypath + "config_aws")

    update_sshconfig(instances)

    print("Don't forget to add this to your ~/.ssh/config file as an include")

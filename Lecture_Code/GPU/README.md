## GPU Examples

Illustrations of using GPUs via three approaches:

1. Directly coding in C/C++ with the NVIDIA APIs
2. Using `RCUDA` to interface with the GPU
3. Using `PyCUDA` to interface with the GPU

Details are provided in each directory.

For using AWS:

+ Log in to your AWS account and go the `EC2` tab

+ Verify that the region is set to `US - Oregon`, `US - Northern California` or `US - East` and if it is not, switch to it to one of those regions.

+ Click on the `AMIs` tab under `Images` in the left-pane of the console

+ Search for image number:
  1. For `US - Oregon` use: `ami-58107568`
  2. For `US - Northern California` use: `ami-367e4e73`
  3. For `US - N. Virginia` use: `ami-3b002d52`

+ Click on the image and then click `Launch`

+ In the left tab select `GPU Instances` and select `g2.2xlarge`, then click `Review and Launch`

+ Check the summary and then click `Launch` (ignore the warnings for now)

+ Select the appropriate keypair for your AWS account and finalize the launch

+ You should receive a message informing you that `Your instance is now launching`

+ Go back to the `EC2` tab under `Services`

+ Check the status of your instance. After a few minutes, click refresh and if it is up and running, ssh into the instance in a similar manner as you did for HW2 i.e., 

    ssh -i mykeypair.pem ec2-user@ec2-XX-XXX-XXX-XXX.us-xxxx-xx.compute.amazonaws.com

+ Once logged in, both `RCUDA` and `PyCUDA` are already installed. To use `RCUDA` simply use `library(RCUDA)` within `R`. To use `PyCUDA`, you will need to invoke python v2.7 by launching `python2.7` instead of just `python`. `PyCUDA` can then be imported in the usual manner.

**Please remember to terminate your AWS instance when you have finished using it!**


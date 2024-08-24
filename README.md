# Panoramic Road Scene Generator
This repo consist of the pre-trained model and workflow to generate panoramic road scenes from a textual prompt

## Generator
Assuming the use of a Nvidia GPU Install cuba enabled torch:
``` pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118```
followed by ```pip install -r requirements.txt```

Please run ``` py generator.py``` to trigger the generation process. you will be prompted in the command line for a generation prompt and tfhe resultant files will be named as ```output.png```


### Server Operation Instructions
After the installation, please proceed to start the fastAPI server. The FastAPI server exposes port 8000 as the endpoint for the generator.

To initiate the generation of the image, please use the ```/predictions``` endpoint using a HTTP POST call with a JSON payload. An example payload is provided below
```
{
  "prompt": "mutiway intersection in the city center"
}
```

this will result a 4096 x 2048 image being generated and encoded into base64 and set back to the client

## Visualizer

This unity based VR visualizer will allow you to place a user in the middle of the generated image. It was created using Unity 2022.3.17f1 and is tested on the Meta Quest line of VR Headsets.

To install the application onto your headset, please install Unity and Visual Studio on your PC and connect the headset.

On your headset please enable developer mode and USB debugging to allow for the installation of non-verified applications

Next, add the project into Unity. The Visualizer is located under the ```RoadGenner``` folder. You may add the entire folder in as a unity project.

Next, within the editor, under Assets, please edit the ```scenegenCall.cs``` to include the IP and port of the above mentioned FastAPI server and save it.

Finally, Build and Run the application on your selected headset.

Within the visualizer, a simple user interface will appear to allow for the user to input their intended scenario

## References

Thanks to the work done by:
 X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,” 2021.
 Their work on the superscaling of the images can be found at https://github.com/xinntao/Real-ESRGAN
 
 S. Orhan and Y. Bastanlar, “Semantic segmentation of outdoor panoramic images,” Signal, Image and Video Processing, vol. 16, pp. 643–650, Aug. 2021, doi: https://doi.org/10.1007/s11760-021-02003-3
 Their dataset of road images dataset CVRG-Pano was used to train the model for the generation of the panoramic images. Their work can be found at https://github.com/semihorhan/semseg-outdoor-pano

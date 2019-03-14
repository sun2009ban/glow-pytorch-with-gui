# glow-pytorch-with-gui
This repository contains the gui for playing with the factorized feature learned from the NinaPro database 5.

The author would like to thank chaiyujin for sharing the code. 

Step1: Clone the code from https://github.com/chaiyujin/glow-pytorch, Don't forget to star the code! :)

Step2: Copy the infer_myo_with_inverse_gui.py to the root dir of the cloned glow-pytorch code.

Step3: Copy the myo.json to the glow-pytorch/hparams/ .

Step4: Download the trained.pkg (~100Mb) at https://pan.baidu.com/s/1jeu4FgukBL6EyYNlZwvrmg , entry code: 5ecf.

Step5: Put the trained.pkg to glow-pytorch/results/myo/ . 

Step6: Run infer_myo_with_inverse_gui.py

Additional Requirements to the original glow-pytorch code:
tkinter

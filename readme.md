# Online learning with CNN
This repository contains all the code for the CS394Z course project, where the Hedge-based and randomnized exponentiated gradient-based online learning framework are implemented.

## Dependency
### System recommendation
It's highly recommended to run this code on a Linux or Mac machine, this code is not tested on Windows machines.
### Python Packages
- Pytorch==2.1.1
- numpy==1.26.0
- torchvision==0.16.1
- matplotlib==3.8.0

Other versions of these packages will most probably work as well.

## Usage
To use this project, follow the steps below:

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
2. **Creat the "History" folder**

   ```bash
   mkdir ./History
3. **Creat/Modify the configuration files under "./Configurations"**
   
   Example configuration files are provided under "./Configurations". A valid configuration file should have a name like "Config_X.json", where "X" is an arbitrary integer.

4. **Run the experiments**
   
   Run the "main.py" file with argument "idx". You can add multiple numbers, and corresponding configuration will be loaded.

   e.g.: The following command runs three experiments, corresponding to file "Config_2.json", "Config_3.json" and "Config_4.json" 

   ```bash
   python ./main.py -idx 2 3 4

5. **Access to the results**
   
   The results including trained model, losses, validation accuracy can be access under "./History/TIME_EXP_INDEX" folder.

   The figures used in the report is drawn using "./draw_figure.ipynb"

6. **Switch between EG and Hegde**
   
   By commenting corresponding lines in "./models/CNN_Online.py" and "./models/MobileNet_Online.py", one can import the class "NN_Online" from either "./models/eg.py" or "./models/hedge.py", i.e. switching between EG and Hedge algorithms.
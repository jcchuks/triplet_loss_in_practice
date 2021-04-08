train_folder = "train"
test_folder = "test"
validate_folder = "val"
test_example_path = "./dataset/chest_xray/test_chest_xray_model"
main_training_sample_path = "dataset/chest_xray/chest_xray"
base_path = test_example_path
best_model_file_name = "best_model.pt"
label_dir = ['NORMAL', 'PNEUMONIA']
mean =  [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch = 7
margin = 1
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
training_and_validation_metric = "t_and_v"
recognition_metric_name = "recog"
result_folder = "./result/"
loss_graph = "loss.png"
metric_graph = "metric.png"
n_epochs = 2

resume_training = False
use_vanilla_model = False
is_live = False

#opensource.trainer constants
best_model_path = "./best_model"
best_model_file_name = "best_model.pt"
validation_file = "validation.txt"
result_folder = './result'
cluster_folder = "./result/clusters/"
folder = ""

#shared_params
shared_params = {}
shared_params["dataset_type"] = "xray"
shared_params["classes"] = 10
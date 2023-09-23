function load_path_hyb_ebm_model(hyb_comp)
%LOAD_PATH_HYB_EBM_MODEL Function to load all path and directory required
%to run the script.
% INPUT: hyb_comp - An integer number used to identify for which component of
% Hybrid-Ebm we are loading the required path. (1 = 1-Component-Lx, 2 =
% 2-Component-Ck)

path_mod_hyb_component = "..\..\..\models\Component-";
path_data_inp_component = "..\..\..\data\input\";

if (hyb_comp == 1)
    path_mod_hyb_component = strcat(path_mod_hyb_component, string(hyb_comp),"-Lx");
    path_data_inp_component = strcat(path_data_inp_component, "Lx");
elseif (hyb_comp == 2)
    path_mod_hyb_component = strcat(path_mod_hyb_component, string(hyb_comp),"-Ck");
    path_data_inp_component = strcat(path_data_inp_component, "Ck");
end

if ~(isfolder(path_mod_hyb_component))
    mkdir(path_mod_hyb_component);
end
addpath(genpath(path_mod_hyb_component));
addpath(genpath(path_data_inp_component));
addpath(genpath("..\..\..\src\lib\utility\"));
addpath(genpath("..\..\..\src\lib\ml\"));
addpath(genpath("..\..\..\src\lib\printer\"));
end
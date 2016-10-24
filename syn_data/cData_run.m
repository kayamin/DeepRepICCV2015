%%%
% live repetition counting system
% Ofir Levy, Lior Wolf
% Tel Aviv University
% 
% Create 30000 train samples and 5000 validations samples in splits of 50
%%%

function cData_run
    close all;
    getrandparamsfunc = @getrandparamwithmotiontype6;
    outdir = '../out/mat';
    totalnTrainSamples = 30000;
    split_size = 50;    
   
    for i=1:(totalnTrainSamples/split_size) % use parfor if you have parallel toolbox
        [all_cFrames,labels,motion_types] = create_cData(split_size,getrandparamsfunc);
        filename = fullfile(outdir,strcat('rep_train_data_', num2str(i)));
        mysave(filename, all_cFrames, labels,motion_types);
        disp(i);
    end
 
    
    nValidSamples = 5000;
    for i=1:(nValidSamples/split_size) % use parfor if you have parallel toolbox
        [all_cFrames,labels,motion_types] = create_cData(split_size,getrandparamsfunc);
        filename = fullfile(outdir,strcat('rep_valid_data_', num2str(i)));
        mysave(filename, all_cFrames, labels,motion_types);
        disp(i);
    end   
end


function mysave(filename, all_cFrames, labels, motion_types)
    if nargin==3
    	save(filename, 'all_cFrames', 'labels');
    elseif nargin==4
        save(filename, 'all_cFrames', 'labels','motion_types');
    else
        error('asdf');
    end
end
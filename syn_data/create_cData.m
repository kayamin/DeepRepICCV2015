function [all_cFrames,labels,motion_types] = create_cData(nSamples,getrandparamsfunc)
    noiselevel = 5;
    nframes = 20;

    labels = randi(8,1,nSamples)-1;
    all_cFrames = cell(1,nSamples*nframes);
    motion_types = zeros(1,nSamples);
    for nsets = 1:nSamples
        fprintf('.');
        if ~mod(nsets,20)
            fprintf('\n');
        end
        [motion_types(nsets),subtype,subtlety,scalefactor] = getrandparamsfunc();
        current_label= labels(nsets);
        cFrames = getonesample(noiselevel,nframes,motion_types(nsets),subtype,subtlety,scalefactor, current_label);
        all_cFrames((1:nframes)+(nsets-1)*nframes) = cFrames;
    end
end
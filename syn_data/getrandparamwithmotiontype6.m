function [motion_type,subtype,subtlety,scalefactor] = getrandparamwithmotiontype6()

scalefactoroptions = [1 1 2 3 4];    
subtletyoptions = [0 2 3];
motion_type_options = [1 2 3 4 5 5 6];

motion_type = motion_type_options(randi(length(motion_type_options),1));
subtype = randi(2,1)-1; %zero or 1, type 1 rotates patches
subtlety = subtletyoptions(randi(length(subtletyoptions),1));
scalefactor = scalefactoroptions(randi(length(scalefactoroptions),1));

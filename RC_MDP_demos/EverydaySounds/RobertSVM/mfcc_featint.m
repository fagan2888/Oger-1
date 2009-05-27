function features = mfcc_featint(file_name, opts)

% Arguments
%   opts.x: 
%   ...
% 
% Return 
%  features: feature vector


% load sound

fprintf('encoding %s \n',file_name); 
[samples, rate] = wavread(file_name);

% compute mfcc per frame
mfcc_frames =  mfcc(samples, rate);

% remove frame swith Nans (leading and trailing 0 samples);
mfcc_frames(:, sum(isnan(mfcc_frames)) > 0) = [];


% feature integration

% first derivative (deltas)

mfcc_delta = diff(mfcc_frames,1,2);

features = [ mean(mfcc_frames,2); ...
             std(mfcc_frames,0,2); ...
             mean(mfcc_delta,2); ...
             std(mfcc_delta,0,2); ];

end



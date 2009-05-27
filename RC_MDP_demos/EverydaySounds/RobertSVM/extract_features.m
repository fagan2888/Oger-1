%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab script for feature calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path with audio files:
DATA_PATH= '/home/holzi/Datasets/closed/data/audio_library/audio_files/';

% path with example lists:
LIST_PATH = '/home/holzi/Datasets/closed/data/general_everyday_sounds/lists/';

% output directory for feature data
OUTPUT_DIR = '../data/'

% example to process
EXAMPLE = 'wind.list';

%EXAMPLES = ["deformation.list", "explosion.list", "friction.list", "pour.list",
%            "whoosh.list", "drip.list", "flow.list", "impact.list",
%            "rolling.list", "wind.list"]

% don't touch the rest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read file with list of soundfiles of current example
listfile = strcat(LIST_PATH,EXAMPLE);
files = textread(listfile,'%s\n');
nfiles = length(files);

% allocate space for features of each file
features = zeros(nfiles, 52);

% calculate all features
for n = 1:nfiles
  curfile =  strcat(DATA_PATH,files{n})
  features(n,:) = mfcc_featint(curfile);
end

% save in matlab style format
savefile = strcat(OUTPUT_DIR,EXAMPLE);
savefile = strcat(savefile(1:end-5),'_mfccint.mat');
save(savefile,'features');


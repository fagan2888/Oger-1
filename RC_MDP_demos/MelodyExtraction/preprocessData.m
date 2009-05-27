function data = preprocessData(inAudioFile, outDataFile)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function data = preprocessData(inAudioFile, outDataFile)
%
% Calculate Audio Features for 
% SVM Meldody Classification
%
% inAudioFile = File Name of .wav file 
%               (e.g. Maroon5_ThisLove.wav)
% outDataFile = Desired File Name of output 
%               (typically Maroon5_ThisLove.csv)
% data = Feature matrix in WEKA .arff format
%        (i.e. Feature 1, Feature 2, . . . ,Label)
%
% graham poliner
% graham@ee.columbia.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nFeat=256;
[mix sr]=wavread(inAudioFile);
d=resample(mix,8000,sr); % resample to 8kHz
[B F T]=specgram(d,1024,8000,1024,944);
B=abs(B(1:nFeat,:)); % power spectrum
B=normftrcols(B,71); % makes zero mean + unit variance over a
                     % local window of 71 frames
                     % (cube root compression - siehe paper)
B=B./max(max(B)); % normalize the data to the maximum
B(find(B<0.0001))=0;
%B(nFeat,:)=60; %Assign arbitrary label for prediction
data=B';
save(outDataFile,'data');
%exit

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = normftrcols(X,W)
%Assign zmuv over a local window

if nargin<2
    W=51;
end

h51 = hann(W);
sh51 = sum(h51);
W2 = floor(W/2);
nr = size(X,1);

Y = 0*X;
for c=1:size(X,2)
    xx = X(:,c);
    mxx = conv(xx,h51)/sh51;
    mxx = mxx(W2+[1:nr]);
    vxx=sqrt(conv(h51, (xx-mxx).^2)/sh51);
    vxx=vxx(W2+[1:nr]);
    vxx=vxx+(vxx==0);
    Y(:,c)=(xx-mxx)./vxx;
end

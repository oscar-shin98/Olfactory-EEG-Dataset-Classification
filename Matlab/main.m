clear;


% 저장할 경로 설정
dataPathConv = fullfile('..', 'Datasets\convert');

% NPMK 툴 경로 설정
NPMKToolPath = 'NPMK';

% 환경설정, EEGLAB Addons 설치 필요
utils.env_setup(NPMKToolPath);

% Raw  데이터 셋 경로 설정
dataPathRaw = fullfile('..', 'Datasets\raw');
subjects = utils.load_dirs(dataPathRaw);

% 전처리기

prep = preprocess(6, 0.5, 70, 1000, 256);

for sub = 1:length(subjects)
    subPath = fullfile(dataPathRaw, subjects(sub).name);

    scents = utils.load_dirs(subPath);
    for sc = 1:length(scents)
        scentPath = fullfile(subPath, scents(sc).name);

        files = utils.load_files(scentPath, 'ns2');

        for file = 1:length(files)
             filePath = fullfile(scentPath, files(file).name);
             try                  
                 eegset = utils.ns2_to_eegset(filePath, dataPathRaw, dataPathConv);
                 prep_eeg = prep.apply(eegset);

                 utils.save_eegset(prep_eeg);
             catch e
                 fprintf("%s\n", e.message);
             end
         end 
    end
end
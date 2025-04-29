% 경로 설정
baseDir = 'your dataset path';  % dataset 폴더 경로로 변경
subjects = dir(baseDir);
subjects = subjects([subjects.isdir] & ~startsWith({subjects.name}, '.'));

for s = 1:length(subjects)
    subjectPath = fullfile(baseDir, subjects(s).name);
    scents = dir(subjectPath);
    scents = scents([scents.isdir] & ~startsWith({scents.name}, '.'));

    for sc = 1:length(scents)
        scentPath = fullfile(subjectPath, scents(sc).name);
        files = dir(fullfile(scentPath, '*.ns2'));

        for f = 1:length(files)
            filePath = fullfile(scentPath, files(f).name);
            try
                %% 1. NS2 파일 로딩
                NSx = openNSx(filePath);

                %% 2. 기준전극 제외 (채널 1~30), µV로 스케일링
		    bit2uV = 0.002;  % nV/bit → 0.002 µV/bit
		    dataEEG = double(NSx.Data(1:30, :)) * bit2uV;
            origSrate = NSx.MetaTags.SamplingFreq;
		    d = designfilt('bandpassiir', ...
              		    'FilterOrder', 6, ...
               		    'HalfPowerFrequency1', 0.5, ...
               		    'HalfPowerFrequency2', 70, ...
               		    'SampleRate', origSrate);  % 여기에 샘플링 주파수 직접 전달
            %% 3. 필터링
		    filteredData = zeros(size(dataEEG));
		    for ch = 1:size(dataEEG, 1)
   			     x = dataEEG(ch,:);
    			    if any(~isfinite(x))
        			    warning("채널 %d에 NaN 또는 Inf 존재, 필터링 생략", ch);
        			    filteredData(ch,:) = x;
    			    else
        			    filteredData(ch,:) = filtfilt(d, x);
    			    end
		    end

            %% 4. EEGLAB 구조체 생성: 필터링된 데이터를 이용하여 초기 EEG 구조체 생성
            EEG = pop_importdata('dataformat', 'array', 'data', filteredData, ...
                                 'srate', origSrate, 'nbchan', 30);
            EEG = eeg_checkset(EEG);

            %% 5. EEGLAB의 pop_resample 함수를 사용한 다운샘플링 (원본: pop_resample)
            downSrate = 256;
            EEG = pop_resample(EEG, downSrate);
            EEG = eeg_checkset(EEG);

            %% 6. 저장 (예: Sub1_A_001.set 형태로 저장)
            saveDir = fullfile('your save path');
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
            filename = sprintf('%s_%s_%03d.set', subjects(s).name, scents(sc).name, f);
            EEG = pop_saveset(EEG, 'filename', filename, 'filepath', saveDir);
            fprintf('Saved: %s\n', filename);

            catch ME
                fprintf('Error with file: %s\n%s\n', filePath, ME.message);
            end
        end
    end
end

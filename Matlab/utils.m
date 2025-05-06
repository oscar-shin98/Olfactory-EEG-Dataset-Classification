classdef utils
    properties
        env_ready = false
    end

    methods(Static, Access = private)
        function addNPMK(npmk_path)
            if nargin < 1 || isempty(npmk_path)
                npmk_path = 'NPMK';
            end
            addpath(npmk_path)
        end
    end 
    
    methods(Static)
%       npmk_path  : NPMK Tool 경로
%       Matlab Addons 'EEGLAB' 설치 필요
        function env_setup(npmk_path)
            utils.addNPMK(npmk_path);
            eeglab()
        end
        
        function dirs=load_dirs(dir_path)
            if ~isfolder(dir_path)
                error("Invalid directory, '%s'.", dir_path);
            end
            
            items = dir(dir_path);
            names = string({items.name});
            is_dir = [items.isdir];
            
            mask = is_dir &  ~startsWith(names, '.') & names~= "." & names~="..";
            dirs = items(mask);
        end

        function files=load_files(dir_path, ext)
            if ~isfolder(dir_path)
                error("Invalid directory, '%s'.", dir_path);
            end

            ext = string(ext);
            if ~startsWith(ext, '.')
                ext = "." + ext;
            end

            pattern = fullfile(dir_path, "*" + ext);
            files_all = dir(pattern);

            files = files_all(~[files_all.isdir]);
        end


        function eegset = ns2_to_eegset(ns2File, inDir, outDir)
            if ~isfile(ns2File)
                error("'%s isn't exists", ns2File);
            end

            if nargin < 2 || isempty(inDir)
                error('undefined outDir.');
            end

            if nargin < 3 || isempty(outDir)
                error('undefined outDir');
            end

            relPath = strrep(fileparts(ns2File), inDir, '');
            if startsWith(relPath, filesep)
                relPath = relPath(2:end);
            end
            relPath = strrep(relPath, '\', '_');
            
            if ~isfolder(outDir)
                mkdir(outDir);
            end

            nsx = openNSx('read', ns2File, 'precision', 'double' ,'uV');

            [~, name, ~] = fileparts(ns2File);

            filename = sprintf('%s_%s.set', relPath, name);

            eegset = eeg_emptyset;
            eegset.data   = nsx.Data(1:30, :);
            eegset.nbchan = size(eegset.data, 1);
            eegset.pnts   = size(eegset.data, 2);
            eegset.srate  = nsx.MetaTags.SamplingFreq;
            eegset.trials = 1;
            eegset.xmin   = 0;
            eegset.xmax   = (eegset.pnts-1) / eegset.srate;
            eegset.setname = name;
            eegset.filename = filename;
            eegset.filepath = outDir;

            labels = { 'CP3', 'FC3', 'FCZ', 'FP1', 'P3', 'FZ', 'CZ', 'FP2', ...
            'P4', 'F8', 'OZ', 'PZ', 'TP8', 'F7', 'F4', 'F3', ...
            'CP4', 'T6', 'FC4', 'T4', 'C3', 'FT7', 'O2', 'C4', ...
            'TP7', 'T5', 'O1', 'T3', 'CPZ', 'FT8' };

            if numel(labels) ~= eegset.nbchan
                 warning('channel(%d), lables(%d) mismatch.', eegset.nbchan, numel(labels));
            end

            for ch = 1:min(eegset.nbchan, numel(labels))
                eegset.chanlocs(ch).labels = labels{ch};
            end

            eegset = eeg_checkset(eegset);
        end
        

        function save_eegset(eeg)
            pop_saveset(eeg, 'filename', eeg.filename, 'filepath', eeg.filepath);
            fprintf('Saved: %s\n', fullfile(eeg.filepath, eeg.filename));  
        end
    end
end
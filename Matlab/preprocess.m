classdef preprocess < handle
    
    properties (Access = private)
        d   % designed filter
        order
        cutLow
        cutHigh
        origin_fs
        target_fs
        isInit = false
    end

    methods
        %
        % oder = 필터 차수
        % cutLow = 하한값
        % cutHigh = 상한값
        % origin_fs = 원본 samplerate
        % target_fs = down sample rate
        function obj = preprocess(order, cutLow, cutHigh, origin_fs, target_fs)
            obj.order = order;
            obj.cutLow = cutLow;
            obj.cutHigh = cutHigh;
            obj.origin_fs = origin_fs;
            obj.target_fs = target_fs;
        end
        
        function initFilter(obj)
            if ~obj.isInit
       		    obj.d = designfilt('bandpassiir', ...
      		    'FilterOrder', obj.order, ...
       		    'HalfPowerFrequency1', obj.cutLow, ...
       		    'HalfPowerFrequency2', obj.cutHigh, ...
       		    'SampleRate', obj.origin_fs);
                
                obj.isInit = true;
            end
        end

        function eeg_out = apply(obj, eeg_in)
            obj.initFilter();

            data = double(eeg_in.data);
            nChan = size(data, 1);

            filtered = zeros(size(data));
            
            % filter
            for ch = 1:nChan
                filtered(ch,:) = filtfilt(obj.d, data(ch,:));
            end
            
            eeg_out = eeg_in;
            eeg_out.data = filtered;
            eeg_out = eeg_checkset(eeg_out);

            % downsample
            if obj.target_fs < obj.origin_fs
                eeg_out = pop_resample(eeg_out, obj.target_fs);
                eeg_out = eeg_checkset(eeg_out);
            end
        end
    end
end

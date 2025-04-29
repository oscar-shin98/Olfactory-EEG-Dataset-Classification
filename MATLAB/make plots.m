EEG = pop_loadset('filename', 'your file(.set)', 'filepath', 'your path');
channelsToPlot = 1:30;
samples = size(EEG.data, 2);
time = (0:samples-1) / EEG.srate;
saveDir = fullfile(pwd, 'folder name');  % 현재 폴더에 저장

if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for ch = channelsToPlot
    f = figure('Visible', 'off');  % 화면에 안 띄우고 백그라운드에서 그림
    plot(time, EEG.data(ch, :));
    title(['EEG Channel ' num2str(ch)]);
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    
    % 저장
    filename = fullfile(saveDir, sprintf('channel_%02d.png', ch));
    saveas(f, filename);
    close(f);
end

disp('✅ 모든 채널 그림 저장 완료!');
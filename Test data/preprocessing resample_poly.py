import numpy as np
import pandas as pd
from scipy import signal
import neo


def process_ns2_file(file_path, output_path, method='resample_poly'):
    """
    .ns2 파일을 읽고 처리하여 CSV 파일로 저장합니다.

    Parameters:
    file_path (str): 입력 .ns2 파일 경로
    output_path (str): 출력 CSV 파일 경로
    method (str): 다운샘플링 방법 ('resample' 또는 'resample_poly')
    """
    # Neo 라이브러리를 사용하여 .ns2 파일 읽기
    reader = neo.io.BlackrockIO(file_path)

    # 모든 채널의 데이터 읽기
    block = reader.read_block()
    sigs = block.segments[0].analogsignals[0]

    # 데이터 배열로 변환
    data = sigs.magnitude

    # 원본 샘플링 속도 (BlackrockIO에서 자동으로 감지)
    orig_fs = float(sigs.sampling_rate)
    print(f"원본 샘플링 속도: {orig_fs} Hz")

    # 31, 32번 채널 제거 (인덱스는 0부터 시작하므로 30, 31)
    channels_to_keep = list(range(30))  # 0부터 29까지의 채널 (1-30번)
    data = data[:, channels_to_keep]

    # 5차 버터워스 밴드패스 필터 설계 (0.5-70Hz)
    nyquist = orig_fs / 2
    low = 0.5 / nyquist
    high = 70 / nyquist
    b, a = signal.butter(5, [low, high], btype='band')

    # 필터 적용
    filtered_data = signal.filtfilt(b, a, data, axis=0)

    # 256Hz로 다운샘플링
    target_fs = 256
    if method == 'resample':
        num_samples = int(filtered_data.shape[0] * target_fs / orig_fs)
        downsampled_data = signal.resample(filtered_data, num_samples, axis=0)
    elif method == 'resample_poly':
        downsampled_data = signal.resample_poly(filtered_data, target_fs, int(orig_fs), axis=0)
    else:
        raise ValueError("method는 'resample' 또는 'resample_poly' 중 하나여야 합니다.")

    # 데이터프레임 생성
    column_names = [f'Channel_{i + 1}' for i in range(len(channels_to_keep))]
    df = pd.DataFrame(downsampled_data, columns=column_names)

    # CSV 파일로 저장
    df.to_csv(output_path, index=False, header=False)
    print(f"처리된 데이터가 {output_path}에 저장되었습니다.")

    return df

# 사용 예시
if __name__ == "__main__":
    input_file = "datafile001.ns2"
    output_file = " processed_eeg_data2.csv"
    processed_data = process_ns2_file(input_file, output_file, method='resample_poly')
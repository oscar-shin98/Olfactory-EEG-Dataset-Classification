import numpy as np
import pandas as pd
from scipy import signal
import neo
import os
import glob
from tqdm import tqdm


def process_ns2_file(file_path, output_path, method='resample_poly'):
    """
    .ns2 파일을 읽고 처리하여 CSV 파일로 저장합니다.

    Parameters:
    file_path (str): 입력 .ns2 파일 경로
    output_path (str): 출력 CSV 파일 경로
    method (str): 다운샘플링 방법 ('resample' 또는 'resample_poly')
    """
    try:
        # Neo 라이브러리를 사용하여 .ns2 파일 읽기
        reader = neo.io.BlackrockIO(file_path)

        # 모든 채널의 데이터 읽기
        block = reader.read_block()
        sigs = block.segments[0].analogsignals[0]

        # 데이터 배열로 변환
        data = sigs.magnitude

        # 원본 샘플링 속도 (BlackrockIO에서 자동으로 감지)
        orig_fs = float(sigs.sampling_rate)

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
        return True
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path}")
        print(f"오류 내용: {str(e)}")
        return False


def batch_process_dataset(dataset_path, output_base_path, method='resample_poly'):
    """
    데이터셋 전체를 일괄 처리합니다.

    Parameters:
    dataset_path (str): 데이터셋 루트 폴더 경로
    output_base_path (str): 출력 CSV 파일을 저장할 기본 경로
    method (str): 다운샘플링 방법
    """
    # 전체 파일 수 계산 (진행 상황 표시용)
    total_files = 0
    for sub_folder in range(1, 12):  # Sub.1 ~ Sub.11
        for condition in range(ord('A'), ord('N')):  # A ~ M
            condition_char = chr(condition)
            pattern = os.path.join(dataset_path, f"Sub. {sub_folder}", condition_char, "datafile*.ns2")
            files = glob.glob(pattern)
            total_files += len(files)

    print(f"총 처리할 파일 수: {total_files}")

    # 진행 상황 표시를 위한 tqdm 초기화
    pbar = tqdm(total=total_files, desc="파일 처리 중")

    # 처리 결과 통계
    success_count = 0
    failed_count = 0

    # 각 폴더의 모든 파일 처리
    for sub_folder in range(1, 12):  # Sub.1 ~ Sub.11
        for condition in range(ord('A'), ord('N')):  # A ~ M
            condition_char = chr(condition)

            # 입력 파일 경로 패턴
            pattern = os.path.join(dataset_path, f"Sub. {sub_folder}", condition_char, "datafile*.ns2")
            input_files = glob.glob(pattern)

            # 폴더가 없으면 건너뛰기
            if not input_files:
                continue

            # 출력 폴더 경로
            output_folder = os.path.join(output_base_path, f"Sub. {sub_folder}", condition_char)

            # 출력 폴더가 없으면 생성
            os.makedirs(output_folder, exist_ok=True)

            # 각 파일 처리
            for input_file in input_files:
                # 파일 이름 추출 (예: datafile001.ns2 -> datafile001)
                file_name = os.path.basename(input_file).split('.')[0]
                output_file = os.path.join(output_folder, f"{file_name}.csv")

                # 파일 처리
                success = process_ns2_file(input_file, output_file, method)

                # 결과 집계
                if success:
                    success_count += 1
                else:
                    failed_count += 1

                # 진행 상황 업데이트
                pbar.update(1)

    # 진행 상황 표시 종료
    pbar.close()

    # 최종 결과 출력
    print(f"처리 완료: 성공 {success_count}, 실패 {failed_count}")


if __name__ == "__main__":
    # 데이터셋 경로와 출력 경로 설정
    dataset_path = "dataset"
    output_base_path = "processed_dataset"

    # 데이터셋 일괄 처리
    batch_process_dataset(dataset_path, output_base_path, method='resample_poly')
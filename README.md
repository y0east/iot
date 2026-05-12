# IoT Servo Tracker

서보모터 기반 팬-틸트 객체 추적 프로젝트입니다. PDF 명세의 핵심 요구사항을 기준으로 라즈베리파이 엣지 제어, RTX 서버 비전 추론, Streamlit 웹 명령 화면, MQTT/ZMQ 통신, 센서 검증, 능동탐색, 안전대기 복구 흐름을 실행 가능한 Python 런타임으로 구성했습니다.

## 핵심 기능

- Streamlit 웹 화면에서 자연어 추적 대상 입력, 추적 시작, 중지, 재검출, 중립각 복귀 명령 생성
- MQTT 명령/상태 패킷 구조와 중복 명령 방지용 `cmd_id` 지원
- ZMQ 영상/추론 패킷 송수신 런타임
- 라즈베리파이 엣지 상태머신: `IDLE`, `SCAN`, `DELAY_COMPENSATION`, `TRACKING`, `SAFE_HOLD`, `LIMITED_RESCAN`, `CENTERING`, `ERROR`
- `Kinit` 연속 검출 기반 능동탐색 확정, 탐색 실패 시 중립각 복귀
- 명령 고유번호 재실행 방지, stale 추론 결과 폐기, thread-safe 런타임 상태 보호
- 픽셀 오차 기반 팬/틸트 PD 제어, deadband, 최대 각속도, 최대 각가속도, PWM 펄스폭 매핑
- ToF/초음파/리밋스위치 기반 단순 임계값 검증과 안전대기 전환
- 안전대기 soft-stop, 동일 대상 재검출, 제한재탐색, timeout 중립각 복귀
- 순환버퍼 기반 지연 보정용 프레임/검출 이력 관리
- WeDetect HTTP 엔드포인트 + Ultralytics YOLO/ByteTrack 계열 production pipeline 경계
- 실제 하드웨어 없이 동작 확인 가능한 시뮬레이션 드라이버와 단위 테스트

## 폴더 구조

```text
.
├── config/                     # 설정 예시
├── docs/                       # 아키텍처와 하드웨어 배선 문서
├── scripts/                    # 로컬 시뮬레이션 스크립트
├── src/iot_servo_tracker/
│   ├── common/                 # 설정, 패킷, 시간 유틸
│   ├── comms/                  # MQTT/ZMQ 통신 어댑터
│   ├── control/                # 상태머신, PD 제어, 센서 검증, 서보 드라이버
│   ├── edge/                   # 라즈베리파이 엣지 런타임
│   ├── server/                 # 추론 서버 런타임과 비전 파이프라인 경계
│   └── web/                    # Streamlit 웹 제어 화면
└── tests/                      # 표준 unittest 기반 테스트
```

## 빠른 실행

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
python3 scripts/simulate_control_loop.py
```

패키지 형태로 설치해서 실행하려면:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[web,edge,server,dev]"
iot-simulate
```

웹 화면은 선택 의존성을 설치한 뒤 실행합니다.

```bash
streamlit run src/iot_servo_tracker/web/app.py
```

## 실제 프로세스 실행

MQTT broker와 ZMQ 통신 주소를 `config/settings.toml`에 맞춘 뒤 실행합니다.

RTX 서버:

```bash
iot-server --config config/settings.toml --serve --production
```

라즈베리파이 엣지:

```bash
iot-edge --config config/settings.toml --run
```

PCA9685 서보와 실제 거리센서를 사용할 때:

```bash
iot-edge --config config/settings.toml --run --hardware-servo --hardware-sensors
```

하드웨어 없이 통신 루프만 확인할 때:

```bash
iot-edge --config config/settings.toml --run --simulated-camera
```

## 실제 장비 연결 시 다음 작업

1. `config/settings.example.toml`을 `config/settings.toml`로 복사하고 MQTT/ZMQ 주소와 서보 한계를 환경에 맞춥니다.
2. `wedetect_endpoint`, `yolo_model`, `tracker`를 RTX 서버 환경에 맞춥니다.
3. 라즈베리파이에서는 `iot-edge --run --hardware-servo --hardware-sensors`로 PCA9685 서보와 ToF/초음파/리밋스위치를 사용합니다.
4. 하드웨어 핀 번호가 다르면 `RaspberryPiSensorReader` 생성 인자를 장비 배선에 맞춥니다.
5. Kp/Kd, `omega_max`, `alpha_max`, `Tpix`, `Ttof`, `Tultra`, `Nh`를 실제 카메라/서보/전원 환경에서 보수적으로 튜닝합니다.

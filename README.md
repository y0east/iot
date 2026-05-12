# IoT Servo Tracker

서보모터 기반 팬-틸트 객체 추적 프로젝트입니다. PDF 명세의 핵심 요구사항을 기준으로 라즈베리파이 엣지 제어, RTX 서버 비전 추론, Streamlit 웹 명령 화면, MQTT/ZMQ 통신, 센서 검증, 안전대기 상태를 분리한 Python 프로젝트 골격을 구성했습니다.

## 핵심 기능

- Streamlit 웹 화면에서 자연어 추적 대상 입력, 추적 시작, 중지, 재검출, 중립각 복귀 명령 생성
- MQTT 명령/상태 패킷 구조와 중복 명령 방지용 `cmd_id` 지원
- ZMQ 영상/추론 패킷을 위한 JSON + binary multipart 헬퍼
- 라즈베리파이 엣지 상태머신: `IDLE`, `SCAN`, `DELAY_COMPENSATION`, `TRACKING`, `SAFE_HOLD`, `LIMITED_RESCAN`, `CENTERING`, `ERROR`
- 픽셀 오차 기반 팬/틸트 PD 제어, deadband, 최대 각속도, 최대 각가속도, PWM 펄스폭 매핑
- ToF/초음파 센서 기반 단순 임계값 검증과 안전대기 전환
- 순환버퍼 기반 지연 보정용 프레임/검출 이력 관리
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

## 실제 장비 연결 시 다음 작업

1. `config/settings.example.toml`을 `config/settings.toml`로 복사하고 MQTT/ZMQ 주소와 서보 한계를 환경에 맞춥니다.
2. `SimulatedServoDriver` 대신 PCA9685 또는 GPIO PWM 드라이버를 구현해 `EdgeRuntime`에 주입합니다.
3. `SimulatedVisionPipeline` 대신 WeDetect 초기 탐지와 YOLO26 + BoT-SORT 또는 ByteTrack 추적기를 연결합니다.
4. ToF, 초음파 센서 값을 `SensorSample`로 변환해 엣지 런타임에 넣습니다.
5. Kp/Kd, `omega_max`, `alpha_max`, `Tpix`, `Ttof`, `Tultra`, `Nh`를 실제 카메라/서보/전원 환경에서 보수적으로 튜닝합니다.

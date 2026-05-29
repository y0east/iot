"""Background voice command listener using SpeechRecognition."""

import threading
import time
import re
from typing import Callable, Optional

from iot_servo_tracker.common.packets import CommandPacket, CommandType


def parse_natural_language_command(text: str) -> Optional[CommandPacket]:
    """
    유연한 자연어 문장에서 의도를 파악하여 명령어 패킷으로 변환합니다.
    - 예: "빨간 컵 좀 찾아주세요", "이제 멈춰", "가운데로 정렬해"
    - 키워드 방식(startswith)이 아닌, 문맥 파싱 방식을 사용합니다.
    """
    text = text.strip()
    if not text:
        return None

    # 0. 나를 봐달라는 특별 요청 (발표 중 시선 집중)
    # 띄어쓰기를 무시하고 매칭하여 인식률을 높임
    text_nospace = text.replace(" ", "")
    if any(word in text_nospace for word in ["저를봐주세요", "나를봐주세요", "나좀봐", "저좀봐주세요", "저봐주세요"]):
        return CommandPacket.create(CommandType.TRACK, query="person")

    # 1. 종료/정지 의도 파악
    if any(word in text for word in ["그만", "멈춰", "정지", "스톱", "종료"]):
        return CommandPacket.create(CommandType.STOP)

    # 2. 중앙 정렬 의도 파악
    if any(word in text for word in ["가운데", "중앙", "정렬", "초기화", "센터"]):
        return CommandPacket.create(CommandType.CENTER)

    # 3. 추적 의도 파악 및 타겟 추출
    # 정규식을 이용해 "~ 찾아주세요", "~ 추적해" 앞의 대상(query)을 유연하게 분리
    match = re.search(r'(.+?)(?:\s*(?:을|를|좀|이라는|라는))?\s*(?:찾아|추적|따라가|잡아|인식|보고)', text)
    if match:
        query = match.group(1).strip()
        # 불필요한 조사나 감탄사 제거 (간단한 필터링)
        query = query.replace("저기", "").replace("혹시", "").replace("제발", "").strip()
        
        # 한국어 쿼리가 입력되었지만 YOLOE 모델이 영어를 더 잘 인식할 경우를 대비해 
        # 자주 쓰는 단어를 영어로 매핑 (필요시 확장)
        # MobileCLIP은 다국어를 지원하므로 그대로 넘겨도 꽤 잘 작동합니다.
        translation_map = {
            "사람": "person",
            "컵": "cup",
            "빨간 컵": "red cup",
            "핸드폰": "cell phone",
            "의자": "chair",
            "얼굴": "face",
            "모니터": "monitor",
        }
        for kr, en in translation_map.items():
            if query == kr:
                query = en
                break

        return CommandPacket.create(CommandType.TRACK, query=query)

    # 4. 짧은 단답형 (예: "사람", "빨간 컵") + "주세요/해줘"
    if any(text.endswith(suffix) for suffix in ["주세요", "해줘", "바래"]):
        # "~주세요" 앞의 모든 단어를 타겟으로 간주
        query = text
        for suffix in ["좀 찾아주세요", "추적해주세요", "주세요", "해줘", "바래"]:
            query = query.replace(suffix, "").strip()
        if query:
            return CommandPacket.create(CommandType.TRACK, query=query)

    return None


class VoiceCommander:
    def __init__(self, command_callback: Callable[[CommandPacket], None]):
        self.command_callback = command_callback
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        try:
            import speech_recognition as sr
        except ImportError as exc:
            print("[Voice] Install optional dependency: pip install '.[voice]'")
            raise exc

        self.recognizer = sr.Recognizer()
        
        # 주변 소음에 맞춰 자동으로 임계값 조절
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000

        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("[Voice] Voice command listener started in background.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _listen_loop(self):
        import speech_recognition as sr
        
        # 기본 마이크로 초기화 (마이크 인덱스를 명시적으로 찾지 않으면 OS 기본값 사용)
        try:
            mic = sr.Microphone()
        except OSError as e:
            print(f"[Voice] Error: No microphone found ({e}). Voice commands disabled.")
            return

        with mic as source:
            print("[Voice] Calibrating microphone for ambient noise... Please wait 2 seconds.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
            print("[Voice] Ready! You can speak now (e.g. '사람 찾아주세요', '이제 멈춰')")

            while self.running:
                try:
                    # phrase_time_limit: 한 마디가 너무 길어지지 않게 제한 (5초)
                    # timeout: 아무 말도 안 하고 있을 때 빠져나오기 위한 제한 (1초)
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5.0)
                except sr.WaitTimeoutError:
                    continue  # 아무 말도 안 함
                except Exception as e:
                    print(f"[Voice] Microphone read error: {e}")
                    time.sleep(1)
                    continue

                if not self.running:
                    break

                # 구글 STT 엔진을 사용해 텍스트로 변환 (인터넷 필요)
                try:
                    # 한국어 우선 인식
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    print(f"[Voice] 인식된 텍스트: '{text}'")

                    # 자연어 파싱
                    packet = parse_natural_language_command(text)
                    if packet:
                        print(f"[Voice] 명령 해석 성공! -> {packet.cmd_type.name} (query: {packet.query})")
                        self.command_callback(packet)
                    else:
                        print("[Voice] 명령어를 이해하지 못했습니다 (의도 파악 실패).")

                except sr.UnknownValueError:
                    # 말소리가 불분명하거나 잡음인 경우
                    pass
                except sr.RequestError as e:
                    print(f"[Voice] Google Speech API request failed (Network error?): {e}")
                except Exception as e:
                    print(f"[Voice] Unexpected error during recognition: {e}")

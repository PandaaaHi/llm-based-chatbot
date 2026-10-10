import subprocess
import os
import signal
import sys
from paddlespeech.cli.asr.infer import ASRExecutor
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

COLORS = {
    'red':      '\033[31m',
    'green':    '\033[32m',
    'white':    '\033[37m',

    'reset':    '\033[0m',
}

MAX_HISTORY_LEN = 20

def cprint(text, color='green'):
    if sys.stdout.isatty():
        code = COLORS.get(color, 'green')
        print(f'{code}{text}{COLORS["reset"]}')
    else:
        print(text)

class VoiceAssistant:
    def __init__(self):
        self.model_name = 'qwen2.5:7b'
        self.temp_audio = 'temp_audio.wav'
        self.output_audio = 'response.wav'

        self.asr = ASRExecutor()

        self.llm = ChatOllama(model=self.model_name, temperature=0.1)
        self.chat_history = [SystemMessage(content=(
            # '你是一个语音助手。'
            # '回答必须用口语化、简短、连贯的纯文本，适合语音朗读，控制在10句话以内。'
            # '禁止使用 Markdown 格式，包括：加粗（**）、斜体、标题（#）、有序列表（1. 2.）、无序列表（- *）、反引号、代码块等。'
            # '不要罗列，直接说人话。'
            'You are a voice assistant, and your name is Alex. '
            'Your answers must be colloquial, concise, and coherent plain text, suitable for voice reading, and limited to 10 sentences. '
            'Do not use any Markdown formatting, including: bold (**), italics, headings (#), ordered lists (1. 2.), unordered lists (- *), backticks, and code blocks. '
            'Do not list items; just speak naturally.'
        ))]

    def record_audio(self):
        input('press Enter to start/stop recording\n')
        cprint('recording now...')

        process = subprocess.Popen(
            ['rec', '-c', '1', '-r', '16000', '-b', '16', self.temp_audio],
            start_new_session=True,
        )

        try:
            input()
        finally:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            process.wait()
        cprint('recording completed\n')

        return self.temp_audio

    def speech_to_text(self, audio_file):
        cprint('transcribing...')
        result = self.asr(
            audio_file=audio_file,
            # model='conformer_talcs',
            # lang='zh_en',
            # codeswitch=True,
            model='transformer_librispeech',
            lang='en',
            device='gpu'
        )
        print(f'transcription result: {result}\n')
        return result

    def _trim_history(self):
        if len(self.chat_history) > MAX_HISTORY_LEN + 1:
            self.chat_history = self.chat_history[:1] + self.chat_history[-MAX_HISTORY_LEN:]

    def get_llm_response(self, user_text):
        cprint('llm thinking...')

        self._trim_history()
        self.chat_history.append(HumanMessage(content=user_text))
        try:
            response = self.llm.invoke(self.chat_history)
            reply = response.content
            self.chat_history.append(AIMessage(content=reply))
            print(f'reply: {reply}\n')
            return reply
        except Exception as e:
            cprint(f'failed to call llm: {e}', 'red')
            self.chat_history.pop()
            return 'service unavailable'

    def text_to_speech(self, text):
        cprint('generating speech...')
        cmd = [
            'paddlespeech', 'tts',
            '--input', text,
            '--am', 'fastspeech2_mix',
            '--voc', 'hifigan_aishell3',
            '--spk_id', '10',
            '--lang', 'mix',
            '--output', self.output_audio,
        ]
        subprocess.run(cmd, check=True)
        cprint('\nspeech generated\n')

        cprint('playing reply...')
        subprocess.run(['aplay', self.output_audio])

    def run(self):
        while True:
            try:
                audio_file = self.record_audio()

                # asr
                user_text = self.speech_to_text(audio_file)
                if not user_text:
                    print('no speech detected')
                    continue

                # llm
                reply = self.get_llm_response(user_text)

                # tts
                self.text_to_speech(reply)

                print('\n' + '-' * 20 + '\n')
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'error: {e}')
                continue

if __name__ == '__main__':
    assistant = VoiceAssistant()
    assistant.run()
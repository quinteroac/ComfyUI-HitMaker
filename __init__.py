from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import moss_music


class MOSSMusicListen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HitMakerMOSSMusicListen",
            display_name="MOSS-Music Listen",
            category="HitMaker",
            description="Ask MOSS-Music a question about a piece of audio: key, tempo, chords, structure, lyrics, instrumentation.",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("question", multiline=True, default="Please give a detailed musical description of this clip."),
                io.Combo.Input("model", options=moss_music.model_names()),
                io.Combo.Input("precision", options=["int8", "nf4", "bf16"], tooltip="bf16 offloads part of the LLM to RAM on 16 GB cards."),
                io.Int.Input("max_new_tokens", default=2048, min=64, max=16384),
                io.Float.Input("temperature", default=0.6, min=0.0, max=2.0, step=0.05, tooltip="0 = greedy."),
                io.Float.Input("repetition_penalty", default=1.05, min=1.0, max=2.0, step=0.01, tooltip="Raise it if transcriptions or chord lists loop."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Keep MOSS-Music in VRAM for the next Listen run. Leave off before running other models."),
            ],
            outputs=[io.String.Output(display_name="answer"), io.String.Output(display_name="thinking")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, question, model, precision, max_new_tokens, temperature, repetition_penalty, seed, keep_loaded):
        thinking, answer = moss_music.ask(audio, question, model, precision, max_new_tokens, temperature, repetition_penalty, seed)
        if not keep_loaded:
            moss_music.unload()
        return io.NodeOutput(answer, thinking, ui={"text": (answer,), "thinking": (thinking,)})


class HitMakerExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MOSSMusicListen]


async def comfy_entrypoint():
    return HitMakerExtension()

def create_ssml(
    lang,
    voice,
    text,
    rate=0,
    pitch=0,
    style=None,
    role=None,
    background_audio=None,
    volume=None,
    fadein=None,
    fadeout=None
):
    sb = []
    sb.append('<speak xmlns="http://www.w3.org/2001/10/synthesis"')
    sb.append(' xmlns:mstts="http://www.w3.org/2001/mstts"')
    sb.append(' xmlns:emo="http://www.w3.org/2009/10/emotionml"')
    sb.append(' version="1.0"')
    sb.append(f' xml:lang="{lang}">')

    if background_audio:
        sb.append(f'<mstts:backgroundaudio src="{background_audio}"')
        if volume: sb.append(f' volume="{volume}"')
        if fadein: sb.append(f' fadein="{fadein}"')
        if fadeout: sb.append(f' fadeout="{fadeout}"')
        sb.append('/>')

    sb.append(f'<voice name="{voice}">')

    if style or role:
        sb.append('<mstts:express-as')
        if style: sb.append(f' style="{style}"')
        if role: sb.append(f' role="{role}"')
        sb.append('>')

        if rate != 0 or pitch != 0:
            sb.append('<prosody')
            if rate != 0: sb.append(f' rate="{rate}%"')
            if pitch != 0: sb.append(f' pitch="{pitch}%"')
            sb.append(f'>{text}</prosody>')
        else:
            sb.append(text)

        sb.append('</mstts:express-as>')
    else:
        if rate != 0 or pitch != 0:
            sb.append('<prosody')
            if rate != 0: sb.append(f' rate="{rate}%"')
            if pitch != 0: sb.append(f' pitch="{pitch}%"')
            sb.append(f'>{text}</prosody>')
        else:
            sb.append(text)

    sb.append('</voice>')
    sb.append('</speak>')

    return "".join(sb)
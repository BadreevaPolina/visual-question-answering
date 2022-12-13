import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def forward_pass():
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    return logits.argmax(-1).item()


def show_image():
    image = Image.open(bottom_image)
    new_image = image.resize((704, 480))
    st.image(new_image)


if __name__ == "__main__":
    st.title("Visual Question Answering")
    image = None

    bottom_image = st.file_uploader('', type='jpg', key=6)
    if bottom_image is not None:
        show_image()

    with st.form(key="question"):
        text = st.text_input(label="Answer your question:")
        submit = st.form_submit_button(label="Compute")
        if submit:
            if text is not None:
                if image is not None:
                    idx = forward_pass()
                    st.write("Predicted answer:", model.config.id2label[idx])
                else:
                    st.write("No image")
            else:
                st.write("No text")

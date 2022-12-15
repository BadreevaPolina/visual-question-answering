import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image


def forward_pass(image, text):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    st.write("Predicted answer:", model.config.id2label[idx])


def save_image(bottom_image):
    image = Image.open(bottom_image)
    new_image = image.resize((704, 480))
    st.image(new_image)
    return new_image


def main():
    st.title("Visual Question Answering")
    image = None

    bottom_image = st.file_uploader('', type='jpg', key=6)
    if bottom_image is not None:
        image = save_image(bottom_image)

    with st.form(key="question"):
        text = st.text_input(label="Answer your question:")
        submit = st.form_submit_button(label="Compute")
        if submit:
            if text is not None:
                if image is not None:
                    forward_pass(image, text)
                else:
                    st.write("No image")
            else:
                st.write("No text")


if __name__ == "__main__":
    main()

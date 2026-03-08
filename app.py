"""
Gradio interface module for DataSense Chat
This file can be used to run just the interface without the main entry point
"""

from main import DataSenseChat, create_gradio_interface, get_api_key


def run_interface():
    """Run just the Gradio interface"""
    api_key = get_api_key()
    chat_app = DataSenseChat(api_key)
    demo = create_gradio_interface(chat_app)
    demo.launch(share=True, debug=False)


if __name__ == "__main__":
    run_interface()
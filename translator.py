def text_to_binary(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    binär_ström = ''.join(format(ord(char), '08b') for char in text)
    return binär_ström

def binary_to_text(binary_data):
    bytes_data = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    text = ''
    for byte in bytes_data:
        if len(byte) == 8: 
            ascii_value = int(''.join(map(str, byte)), 2)
            text += chr(ascii_value)
    return text
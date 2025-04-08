import os
import socket
import struct


class ImageClient:

    def __init__(self, server_address="/tmp/unix_socket_server", image_size=480 * 640 * 3, packet_size=8192):
        self.SERVER_ADDRESS = server_address
        # 640 * 480 * 3 ints for the image 
        # + 1 float for reward = number of completed levels + (diagonal length - distance to goal) / diagonal length
        self.frame_data_fmt_string = f"{image_size}cf"
        self.buffer_size = struct.calcsize(self.frame_data_fmt_string)
        self.server_socket = None
        self.conn = None
        self.server_socket, self.conn = self.start_server()
        self.image_size = image_size
        self.packet_size = packet_size
        self.frame_buffer = bytearray(self.buffer_size)
        self.index = 0

    def start_server(self):
        if os.path.exists(self.SERVER_ADDRESS):
            os.remove(self.SERVER_ADDRESS)

        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(self.SERVER_ADDRESS)
        server_socket.listen(1)
        print("Started server, waiting for connection!")
        conn, _ = server_socket.accept()
        print("Connected to a client")
        return server_socket, conn

    def readable_move(self, move: list) -> list:
        keys = ["up", "down", "left", "right", "z", "x", 'c']
        data = []
        for mv, key in zip(move, keys):
            if mv > 0.5:
                data.append(key)
        return data
    
    def process_frame_and_send_move(self, select_action, optimize_model, update_model_params, memory):
        prev_state = None
        prev_move = None
        prev_reward = None

        while True:
            received_data = self.conn.recv(self.packet_size)
            if received_data[:3] == b"END":
                self.conn.sendall(bytes(1)) # send ack
                assert prev_state is not None, "prev_state should not be None at session end, but it is"
                memory.push(prev_state, prev_move, None, prev_reward)
                prev_state = None
                prev_move = None
                prev_reward = None
                optimize_model()
                update_model_params()
                return
            else:
                bytes_remaining = self.buffer_size - self.index
                bytes_needed = min(bytes_remaining, len(received_data))
                self.frame_buffer[self.index:self.index + bytes_needed] = received_data[:bytes_needed]
                self.index += bytes_needed
                if self.index == self.buffer_size:
                    frame = self.frame_buffer[:self.image_size]
                    if prev_state is not None:
                        memory.push(prev_state, prev_move, frame, prev_reward)
                        optimize_model()
                        update_model_params()
                    
                    prev_state = frame
                    prev_move = select_action(frame).item()
                    prev_reward = struct.unpack('f', self.frame_buffer[self.image_size:])

                    self.conn.sendall(prev_move.to_bytes())
                    # update frame buffer for extra data received
                    index = len(received_data) - bytes_needed
                    self.frame_buffer[:index] = received_data[bytes_needed:]
                    self.frame_buffer[index:] = bytes(self.buffer_size - index)
                    self.index = index


    def test_moves(self):
        moves = [0] + list(2 ** i for i in range(7))
        move_index = 0
        while True:
            buf = self.conn.recv(self.buffer_size)
            move_code = moves[move_index]
            move_index = (move_index + 1) % len(moves)
            # move_code = 8
            self.conn.sendall(move_code.to_bytes())

    def __del__(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    while True:
        try:
            client = ImageClient()
            client.test_moves()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)

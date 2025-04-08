import os
import socket
import struct
import json


class NEATAIClient:

    def __init__(self, format_string='16i6f????', server_address="/tmp/unix_socket_server"):
        self.format_string = format_string
        self.session_feedback_format_string = "2i2f?"
        self.size = struct.calcsize(format_string)
        self.player_state_size = struct.calcsize(self.session_feedback_format_string)
        self.SERVER_ADDRESS = server_address
        self.server_socket = None
        self.conn = None
        self.server_socket, self.conn = self.start_server()
        self.input_data_names = [
            "dist_e",
            "disttype_e",
            "dist_ne",
            "disttype_ne",
            "dist_n",
            "disttype_n",
            "dist_nw",
            "disttype_nw",
            "dist_w",
            "disttype_w",
            "dist_sw",
            "disttype_sw",
            "dist_s",
            "disttype_s",
            "dist_se",
            "disttype_se",
            "pos_x",
            "pos_y",
            "speed_x",
            "speed_y",
            "stamina",
            "distanceToGoal",
            "has_dash",
            "on_ground",
            "is_swim",
            "is_climbing",
        ]
        # output_node_names = [
        #     "up",
        #     "down",
        #     "left",
        #     "right",
        #     "z",
        #     "x",
        #     "c"
        # ]
        # nodes = {
        #     -i: inn for i, inn in enumerate(input_node_names, start=1)
        # }
        # for i, node in enumerate(output_node_names, start=1):
        #     nodes[i] = node

    def _encode_agent_move(self, move) -> int:
        # up down left right z x c
        return sum(1 << idx if pos > 0.5 else 0 for idx, pos in enumerate(move))
    
    def _decode_session_feedback(self, buffer) -> dict:
        deserialized_data = struct.unpack(self.session_feedback_format_string, buffer)
        return {
            "num_completed_levels": deserialized_data[0],
            "elapsed_seconds": deserialized_data[1],
            "level_diagonal_length": deserialized_data[2],
            "distance_to_goal": deserialized_data[3],
            "staying_still_penalty": 0.25 if deserialized_data[4] else 0
        }

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
    
    def _print_neat_data(self, neat_data):
        readable_data = {}
        for name, data in zip(self.input_data_names, neat_data):
            readable_data[name] = data
        print(json.dumps(readable_data, indent=2))

    def process_frame_and_send_move(self, net_activate):
        while True:
            buf = self.conn.recv(self.size)
            if len(buf) < self.size and buf.decode() == "END":
                self.conn.sendall(bytes(1))
                buf = self.conn.recv(self.player_state_size)
                self.conn.sendall(bytes(1))
                final_player_state = self._decode_session_feedback(buf)
                return final_player_state
            else:
                neat_data = struct.unpack(self.format_string, buf)
                # self._print_neat_data(neat_data)
                move = net_activate(neat_data)
                # print(f"Sending move {self.readable_move(move)}")
                move_code = self._encode_agent_move(move)
                self.conn.sendall(move_code.to_bytes())

    def test_moves(self):
        moves = list(2 ** i for i in range(7))
        moves.append(0)
        move_index = 0
        while True:
            buf = self.conn.recv(self.size)
            # neat_data = struct.unpack(self.format_string, buf)
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
            client = NEATAIClient()
            client.test_moves()
        except KeyboardInterrupt:
            break
        except:
            pass

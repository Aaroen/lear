import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("井字棋游戏")
        self.current_player = "X"
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        """创建游戏界面"""
        for row in range(3):
            for col in range(3):
                button = tk.Button(self.root, text="", font=('Arial', 30), width=5, height=2,
                                 command=lambda r=row, c=col: self.on_button_click(r, c))
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

        reset_button = tk.Button(self.root, text="重新开始", command=self.reset_game)
        reset_button.grid(row=3, column=0, columnspan=3, sticky="we")

    def on_button_click(self, row, col):
        """处理按钮点击事件"""
        if self.buttons[row][col]['text'] == "" and not self.check_winner():
            self.buttons[row][col]['text'] = self.current_player
            if self.check_winner():
                messagebox.showinfo("游戏结束", f"玩家 {self.current_player} 获胜！")
                self.reset_game()
            elif self.is_board_full():
                messagebox.showinfo("游戏结束", "平局！")
                self.reset_game()
            else:
                self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        """检查胜利条件"""
        # 检查行
        for row in range(3):
            if self.buttons[row][0]['text'] == self.buttons[row][1]['text'] == self.buttons[row][2]['text'] != "":
                return True
        
        # 检查列
        for col in range(3):
            if self.buttons[0][col]['text'] == self.buttons[1][col]['text'] == self.buttons[2][col]['text'] != "":
                return True
        
        # 检查对角线
        if self.buttons[0][0]['text'] == self.buttons[1][1]['text'] == self.buttons[2][2]['text'] != "":
            return True
        if self.buttons[0][2]['text'] == self.buttons[1][1]['text'] == self.buttons[2][0]['text'] != "":
            return True
        
        return False

    def is_board_full(self):
        """检查棋盘是否已满"""
        for row in range(3):
            for col in range(3):
                if self.buttons[row][col]['text'] == "":
                    return False
        return True

    def reset_game(self):
        """重置游戏"""
        for row in range(3):
            for col in range(3):
                self.buttons[row][col]['text'] = ""
        self.current_player = "X"

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()

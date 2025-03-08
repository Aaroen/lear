class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        a = 0
        b = 0
        for move in moves:
            if move == 'R':
                a -=1
            elif move == 'L':
                a +=1
            elif move == 'U':
                b +=1
            elif move == 'D':
                b -=1
        
        if a==0 and b==0:
            return True
        else:
            return False
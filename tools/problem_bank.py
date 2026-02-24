"""
tools/problem_bank.py – Curated problem sets for Python, Java, and DSA (Beginner to Advanced).
"""

CURATED_PROBLEMS = {
    "Python": [
        {
            "id": "py_beg_1",
            "title": "FizzBuzz",
            "difficulty": "Easy",
            "level": "Beginner",
            "description": "Write a program that prints numbers from 1 to n. For multiples of 3, print 'Fizz'; for multiples of 5, print 'Buzz'; for both, print 'FizzBuzz'.",
            "examples": [{"input": "n=15", "output": "['1','2','Fizz',...]"}],
            "constraints": ["1 <= n <= 100"],
            "solution_template": "def fizzBuzz(n: int) -> list[str]:\n    # Your code here\n    pass",
            "test_cases": [{"input": "15", "expected": "['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']"}],
            "max_score": 10,
            "time_limit": 10
        },
        {
            "id": "py_int_1",
            "title": "Longest Substring Without Repeating Characters",
            "difficulty": "Medium",
            "level": "Intermediate",
            "description": "Given a string s, find the length of the longest substring without repeating characters.",
            "examples": [{"input": "s = 'abcabcbb'", "output": "3"}],
            "constraints": ["0 <= s.length <= 5 * 10^4"],
            "solution_template": "def lengthOfLongestSubstring(s: str) -> int:\n    # Your code here\n    pass",
            "test_cases": [{"input": "abcabcbb", "expected": "3"}],
            "max_score": 25,
            "time_limit": 20
        },
        {
            "id": "py_adv_1",
            "title": "Regular Expression Matching",
            "difficulty": "Hard",
            "level": "Advanced",
            "description": "Implement regular expression matching with support for '.' and '*'.",
            "examples": [{"input": "s = 'aa', p = 'a*'", "output": "true"}],
            "constraints": ["s and p contain only lowercase English letters."],
            "solution_template": "def isMatch(s: str, p: str) -> bool:\n    # Your code here\n    pass",
            "test_cases": [{"input": "aa, a*", "expected": "True"}],
            "max_score": 50,
            "time_limit": 45
        }
    ],
    "Java": [
        {
            "id": "java_beg_1",
            "title": "Two Sum",
            "difficulty": "Easy",
            "level": "Beginner",
            "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "examples": [{"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"}],
            "constraints": ["2 <= nums.length <= 10^4"],
            "solution_template": "class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        // Your code here\n    }\n}",
            "test_cases": [{"input": "[2,7,11,15], 9", "expected": "[0,1]"}],
            "max_score": 15,
            "time_limit": 15
        },
        {
            "id": "java_int_1",
            "title": "Reverse Linked List",
            "difficulty": "Medium",
            "level": "Intermediate",
            "description": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
            "examples": [{"input": "[1,2,3,4,5]", "output": "[5,4,3,2,1]"}],
            "constraints": ["The number of nodes in the list is the range [0, 5000]."],
            "solution_template": "/**\n * Definition for singly-linked list.\n * public class ListNode {\n *     int val;\n *     ListNode next;\n *     ListNode() {}\n *     ListNode(int val) { this.val = val; }\n *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }\n * }\n */\nclass Solution {\n    public ListNode reverseList(ListNode head) {\n        // Your code here\n    }\n}",
            "test_cases": [{"input": "[1,2,3,4,5]", "expected": "[5,4,3,2,1]"}],
            "max_score": 25,
            "time_limit": 25
        },
        {
            "id": "java_adv_1",
            "title": "Median of Two Sorted Arrays",
            "difficulty": "Hard",
            "level": "Advanced",
            "description": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
            "examples": [{"input": "nums1 = [1,3], nums2 = [2]", "output": "2.0"}],
            "constraints": ["O(log(m+n)) time complexity required."],
            "solution_template": "class Solution {\n    public double findMedianSortedArrays(int[] nums1, int[] nums2) {\n        // Your code here\n    }\n}",
            "test_cases": [{"input": "[1,3], [2]", "expected": "2.0"}],
            "max_score": 45,
            "time_limit": 40
        }
    ],
    "DSA": [
        {
            "id": "dsa_beg_1",
            "title": "Valid Anagram",
            "difficulty": "Easy",
            "level": "Beginner",
            "description": "Given two strings s and t, return true if t is an anagram of s, and false otherwise.",
            "examples": [{"input": "s = 'anagram', t = 'nagaram'", "output": "true"}],
            "constraints": ["1 <= s.length, t.length <= 5 * 10^4"],
            "solution_template": "def isAnagram(s: str, t: str) -> bool:\n    # Your code here\n    pass",
            "test_cases": [{"input": "anagram, nagaram", "expected": "True"}],
            "max_score": 10,
            "time_limit": 10
        },
        {
            "id": "dsa_int_1",
            "title": "Binary Tree Level Order Traversal",
            "difficulty": "Medium",
            "level": "Intermediate",
            "description": "Given the root of a binary tree, return the level order traversal of its nodes' values.",
            "examples": [{"input": "root = [3,9,20,null,null,15,7]", "output": "[[3],[9,20],[15,7]]"}],
            "constraints": ["The number of nodes in the tree is in the range [0, 2000]."],
            "solution_template": "def levelOrder(root: Optional[TreeNode]) -> list[list[int]]:\n    # Your code here\n    pass",
            "test_cases": [{"input": "[3,9,20,None,None,15,7]", "expected": "[[3],[9,20],[15,7]]"}],
            "max_score": 30,
            "time_limit": 30
        },
        {
            "id": "dsa_adv_1",
            "title": "Word Search II",
            "difficulty": "Hard",
            "level": "Advanced",
            "description": "Given an m x n board of characters and a list of strings words, return all words on the board.",
            "examples": [{"input": "board = [['o','a','a','n'],['e','t','a','e'],['i','h','k','r'],['i','f','l','v']], words = ['oath','pea','eat','rain']", "output": "['eat','oath']"}],
            "constraints": ["m == board.length", "n == board[i].length", "1 <= words.length <= 3 * 10^4"],
            "solution_template": "def findWords(board: list[list[str]], words: list[str]) -> list[str]:\n    # Your code here\n    pass",
            "test_cases": [{"input": "board, words", "expected": "['eat','oath']"}],
            "max_score": 55,
            "time_limit": 50
        }
    ]
}

def get_curated_problems(category: str):
    return CURATED_PROBLEMS.get(category, [])

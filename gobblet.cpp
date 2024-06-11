// Retrograde analysis of perfect game play for Gobblet Gobblers and similar
// 3x3 Tic Tac Toe variants

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <cstdio>

// A game state is a 54-bit bitboard, 6 bits for each of 3x3=9 squares, 2 bits
// for each piece size (332211), indicating:
//   00 = no piece of that size,
//   01 = piece of player to move, or
//   10 = opponent's piece.
typedef std::uint64_t State;

// A move indicates the starting and ending square in {0..8} when moving a
// piece already on the board, or start in {-1, -2, -3} indicates playing a
// new piece of the corresponding (negated) size.
struct Move { int start, end; };

struct Arena
{
    char *beg{};
    char *end{};

    Arena(ptrdiff_t cap) : beg{new char[cap]}, end{beg+cap} {}
    // NOTE: implementing ~Arena requires also tracking ownership of the
    // underlying region, i.e. scratch copies don't free

    template<typename T, typename ...A>
    T *make(ptrdiff_t count = 1, A ...args)
    {
        ptrdiff_t size = sizeof(T);
        ptrdiff_t pad  = (size_t)end & (alignof(T) - 1);
        ptrdiff_t avail = end - beg;
        if (count >= (avail - pad)/size) {
            __builtin_trap();  // DELETEME
            throw std::bad_alloc();
        }
        T *r = (T *)(end -= pad + size*count);
        for (ptrdiff_t i = 0; i < count; i++) {
            new (r+i) T(args...);
        }
        return r;
    }
};

// SplitMix64
static std::uint64_t hash(State h)
{
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9u;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebu;
    h ^= h >> 31;
    return h;
}

struct MoveSet
{
    struct Node
    {
        Node* child[4]{};
        Node* next{};
        State s{};
        Move  m{};
    };

    Node     *head{};
    ptrdiff_t len{};
    Node    **tail = &head;
    Node     *root{};

    bool insert(Arena *a, State s, Move m = {0, 0})
    {
        Node **n = &root;
        for (std::uint64_t h = hash(s); *n; h <<= 2) {
            if (s == (*n)->s) {
                return 0;
            }
            n = &(*n)->child[h>>62];
        }
        *n = a->make<Node>();
        (*n)->s = s;
        (*n)->m = m;
        *tail = *n;
        tail = &(*n)->next;
        len++;
        return 1;
    }

    // TODO: disable mutation after copy
};

struct Queue
{
    struct Node
    {
        Node *next;
        State s;
    };

    Node  *head{};
    Node **tail = &head;
    Node  *free{};

    bool empty()
    {
        return tail == &head;
    }

    State front()
    {
        return head->s;
    }

    void pop()
    {
        Node *n = head;
        head = n->next;
        if (!head) {
            tail = &head;
        }
        n->next = free;
        free = n;
    }

    void push(Arena *a, State s)
    {
        Node *n = free;
        if (n) {
            free = n->next;
            n->next = 0;
        } else {
            n = a->make<Node>();
        }
        n->s = s;
        *tail = n;
        tail = &n->next;
    }

    // TODO: delete copy operator and constructor
};

class Game
{
    // Define rule variations:
    int num_sizes = 3; // number of piece sizes (<= 3)
    int num_per_size = 2; // number of pieces of each size (per player)
    bool allow_move = true; // whether pieces already on the board may be moved

    // Store all possible game states using MSI hash map (ref. Chris Wellons
    // https://nullprogram.com/blog/2022/08/08/) from each 54-bit bitboard key
    // to its win/loss/draw value packed in the upper 10 bits.
    State* hash_map{};
    const int HASH_EXP = 29;
    const std::size_t HASH_MASK = (1ull << HASH_EXP) - 1;
    const State STATE_EMPTY = 0x3; // 0x0 is the (valid) initial board state
    const State STATE_MASK = (1ull << 54) - 1;

    // Return pointer to hash map entry for given game state.
    State* lookup(State s)
    {
        std::uint64_t h = hash(s);
        std::size_t step = (h >> (64 - HASH_EXP)) | 1;
        for (std::size_t i = h;;)
        {
            i = (i + step) & HASH_MASK;
            if (hash_map[i] == STATE_EMPTY || (hash_map[i] & STATE_MASK) == s)
            {
                return &hash_map[i];
            }
        }
    }

    // First step of retrograde analysis: breadth-first search all states from
    // initial board, returning queue of solved (game-over won or lost) states.
    Queue *search(State s0, Arena *a, Arena scratch)
    {
        std::cout << "Searching... " << std::flush;
        int count = 0;
        Queue *solved = a->make<Queue>();
        Queue q;
        q.push(&scratch, s0);
        *lookup(s0) = s0;
        while (!q.empty())
        {
            State current = q.front();
            q.pop();
            ++count;
            int value = get_terminal_value(current);
            if (value != 0)
            {
                // Queue game-over state as win or loss in 0 moves.
                *lookup(current) = current | pack(value, 0);
                solved->push(a, current);
            }
            else
            {
                // Mark all other states as tentative draw (value 0), recording
                // number of possible (winning) moves.
                Arena temp = *a;
                MoveSet moves = get_moves(current, &temp);
                *lookup(current) = current | pack(0, moves.len);
                for (MoveSet::Node *n = moves.head; n; n = n->next)
                {
                    Move m = n->m;
                    State next = canonical(swap(move(current, m)));

                    // Trade time for extra lookup of next state for memory
                    // from duplicate queue entries.
                    State* next_ptr = lookup(next);
                    if (*next_ptr == STATE_EMPTY)
                    {
                        q.push(&scratch, next);
                        *next_ptr = next;
                    }
                }
            }
        }
        std::cout << "found " << count << " states." << std::endl;
        return solved;
    }

    // Second step of retrograde analysis: work backward breadth-first from
    // initial queue of game-over states, propagating solved win/loss values
    // and incrementing depth to win.
    void solve(Queue *solved, Arena *perm, Arena scratch)
    {
        std::cout << "Solving... " << std::flush;
        int count = 0;
        while (!solved->empty())
        {
            State current = solved->front();
            solved->pop();
            ++count;
            Arena temp = scratch;
            MoveSet unmoves = get_unmoves(current, &temp);
            for (MoveSet::Node *n = unmoves.head; n; n = n->next)
            {
                State prev = n->s;
                State* prev_ptr = lookup(prev);
                if (unpack_value(*prev_ptr) == 0)
                {
                    State* current_ptr = lookup(current);
                    if (unpack_value(*current_ptr) == 1)
                    {
                        // Losing move for previous player; decrement number of
                        // possible winning moves.
                        std::size_t moves = unpack_moves(*prev_ptr);
                        if (--moves != 0)
                        {
                            *prev_ptr = prev | pack(0, moves);
                        }
                        else
                        {
                            // No winning moves; record loss for previous
                            // player and queue solved state.
                            *prev_ptr = prev |
                                pack(-1, unpack_moves(*current_ptr) + 1);
                            solved->push(perm, prev);
                        }
                    }
                    else
                    {
                        // At least one winning move; record win and queue.
                        *prev_ptr = prev |
                            pack(1, unpack_moves(*current_ptr) + 1);
                        solved->push(perm, prev);
                    }
                }
            }
        }
        std::cout << "solved " << count << " win/loss states." << std::endl;
    }

public:
    // Pack win/loss/draw key value into upper 10 bits of state:
    //   01######## = win for current player in # moves
    //   10######## = draw with -(#+1) potential winning moves (2's complement)
    //   11######## = loss in -(#+1) moves (2's complement)
    State pack(int value, std::size_t moves)
    {
        return (value == -1 ? 0 : 1ull << 62) ^
            ((value == 1 ? moves : 0 - (moves + 1)) << 54);
    }

    int unpack_value(State s)
    {
        return 2 - static_cast<int>(s >> 62);
    }

    std::size_t unpack_moves(State s)
    {
        std::int64_t moves = static_cast<std::int64_t>(s << 2) >> 56; // C++20
        return moves < 0 ? (-moves - 1) : moves;
    }

    // With this key-value encoding, the best move maximizes next game state.
    Move best_move(State s, Arena a)
    {
        Move best{};
        State max_next = 0;
        MoveSet moves = get_moves(s, &a);
        for (MoveSet::Node *n = moves.head; n; n = n->next)
        {
            Move m = n->m;
            State next = *lookup(canonical(swap(move(s, m))));
            if (next > max_next)
            {
                max_next = next;
                best = m;
            }
        }
        return best;
    }

    // Initialize and solve game for these rules, loading from disk for speed.
    Game(int num_sizes, int num_per_size, bool allow_move, Arena *a, Arena scratch)
    {
        init(num_sizes, num_per_size, allow_move, a, scratch);
    }

    void init(int num_sizes, int num_per_size, bool allow_move, Arena *a, Arena scratch)
    {
        this->num_sizes = num_sizes;
        this->num_per_size = num_per_size;
        this->allow_move = allow_move;
        hash_map = a->make<State>(HASH_MASK + 1, STATE_EMPTY);

        // Use cached evaluation of game states if available.
        std::string filename = "gobblet_" + std::to_string(num_sizes) + "_" +
            std::to_string(num_per_size) + "_" +
            std::to_string(allow_move) + ".dat";
        std::FILE* fid = 0;//std::fopen(filename.c_str(), "rb");
        if (fid != 0)
        {
            std::cout << "Loading from " << filename << std::endl;
            std::fread(&hash_map[0], sizeof(State), HASH_MASK+1, fid);
        }
        else
        {
            // Cache not found; solve game and save for future re-use.
            Arena temp = *a;
            solve(search(0, &temp, scratch), &temp, scratch);
            fid = std::fopen(filename.c_str(), "wb");
            #if 0
            for (size_t i = 0; i < HASH_MASK+1; i++) {
                std::fwrite(hash_map+i, sizeof(State), 1, fid);
            }
            #endif
        }
        std::fclose(fid);
    }

    // Almost all of the above is general to retrograde analysis of any
    // symmetric two-player game of perfect information. Now implement Gobblet.

    // Make move as current player.
    State move(State s, Move m)
    {
        int size = -m.start;
        if (m.start >= 0)
        {
            // Remove piece already on the board.
            unsigned pieces = (s >> (6 * m.start)) & 0x3f;
            for (size = 0; pieces != 0; ++size, pieces >>= 2);
            s ^= 0x1ull << (6 * m.start + 2 * (size - 1));
        }
        // Place or move piece to new square.
        return s ^ (0x1ull << (6 * m.end + 2 * (size - 1)));
    }

    // Swap colors so it's always X to move, reducing state space by ~1/2.
    State swap(State s)
    {
        return ((s & 0x2aaaaaaaaaaaaau) >> 1) | ((s & 0x15555555555555u) << 1);
    }

    // Rotate/reflect board to minimum representation, reducing space by ~7/8.
    State canonical(State s)
    {
        State min_s = s;
        s = flipud(s);        min_s = s < min_s ? s : min_s;
        s = antitranspose(s); min_s = s < min_s ? s : min_s;
        s = flipud(s);        min_s = s < min_s ? s : min_s;
        s = antitranspose(s); min_s = s < min_s ? s : min_s;
        s = flipud(s);        min_s = s < min_s ? s : min_s;
        s = antitranspose(s); min_s = s < min_s ? s : min_s;
        s = flipud(s);        min_s = s < min_s ? s : min_s;
        return min_s;
    }

    // Mirror board vertically, swapping top and bottom rows.
    State flipud(State s)
    {
        return ((s << 36) & 0x3ffff000000000) | (s & 0xffffc0000) | (s >> 36);
    }

    // Mirror board about off-diagonal.
    State antitranspose(State s)
    {
        return ((s << 48) & 0x3f000000000000) | ((s << 24) & 0xfc0fc0000000) |
            (s & 0x3f03f03f000) | ((s >> 24) & 0xfc0fc0) | (s >> 48);
    }

    // Return value for current player if game over, otherwise 0.
    int get_terminal_value(State s)
    {
        const int lines[8][3] = {
            {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // rows
            {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // columns
            {0, 4, 8}, {2, 4, 6}             // diagonals
        };
        int value = 0;
        for (auto& line : lines)
        {
            unsigned line_winner = 0;
            for (int square = 0; square < 3; ++square)
            {
                unsigned pieces = (s >> (6 * line[square])) & 0x3f;
                for (; pieces > 0x3; pieces >>= 2);
                if (pieces == 0)
                {
                    line_winner = 0;
                    break;
                }
                if (line_winner == 0)
                {
                    line_winner = pieces;
                }
                else if (pieces != line_winner)
                {
                    line_winner = 0;
                    break;
                }
            }

            // You win if your opponent "uncovers" your existing 3-in-a-row,
            // even if they create their own 3-in-a-row in the same move.
            if (line_winner == 1)
            {
                value = 1;
                break;
            }
            else if (line_winner == 2)
            {
                value = -1;
            }
        }
        return value;
    }

    // Return possible moves for current player, ignoring whether
    // get_terminal_value(s) != 0.
    MoveSet get_moves(State s, Arena *a)
    {
        int played[3] = { 0 };
        MoveSet states;

        // Try to move pieces already on the board.
        for (int start = 0; start < 9; ++start)
        {
            unsigned pieces = (s >> (6 * start)) & 0x3f;
            unsigned owner = 0;
            int size = 0;
            for (; pieces != 0; ++size, pieces >>= 2)
            {
                owner = pieces & 0x3;
                if (owner == 1)
                {
                    // Track total number of each size for playing new pieces.
                    ++played[size];
                }
            }
            if (allow_move && owner == 1)
            {
                for (int end = 0; end < 9; ++end)
                {
                    pieces = (s >> (6 * end)) & 0x3f;
                    if (0x1u << (2 * (size - 1)) > pieces)
                    {
                        Move m{start, end};
                        State next = canonical(swap(move(s, m)));
                        // Only list moves distinct up to symmetry.
                        states.insert(a, next, m);
                    }
                }
            }
        }

        // Try to play new pieces.
        for (int size = 1; size <= num_sizes; ++size)
        {
            if (played[size - 1] < num_per_size)
            {
                for (int end = 0; end < 9; ++end)
                {
                    unsigned pieces = (s >> (6 * end)) & 0x3f;
                    if (0x1u << (2 * (size - 1)) > pieces)
                    {
                        Move m{-size, end};
                        State next = canonical(swap(move(s, m)));
                        states.insert(a, next, m);
                    }
                }
            }
        }
        return states;
    }

    // Return list of "unmoves," or previous states leading to given state.
    MoveSet get_unmoves(State s, Arena *a)
    {
        MoveSet unmoves;
        s = swap(s);
        for (int end = 0; end < 9; ++end)
        {
            unsigned pieces = (s >> (6 * end)) & 0x3f;
            unsigned owner = 0;
            int size = 0;
            for (; pieces != 0; ++size, pieces >>= 2)
            {
                owner = pieces & 0x3;
            }
            if (owner == 1)
            {
                if (allow_move)
                {
                    // Try to (un)move piece to previous square.
                    for (int start = 0; start < 9; ++start)
                    {
                        pieces = (s >> (6 * start)) & 0x3f;
                        if (0x1u << (2 * (size - 1)) > pieces)
                        {
                            State prev = move(s, Move{end, start});

                            // Verify that the game wasn't already over.
                            if (get_terminal_value(prev) == 0)
                            {
                                unmoves.insert(a, canonical(prev));
                            }
                        }
                    }
                }

                // Try to (un)play (i.e., remove) new piece.
                State prev = move(s, Move{-size, end});
                if (get_terminal_value(prev) == 0)
                {
                    unmoves.insert(a, canonical(prev));
                }
            }
        }
        return unmoves;
    }

    // Display current game state (hiding any covered pieces).
    void show(State s)
    {
        for (int row = 0; row < 3; ++row)
        {
            std::cout << "      |      |" << std::endl;
            for (int col = 0; col < 3; ++col)
            {
                unsigned pieces = (s >> (6 * (3 * row + col))) & 0x3f;
                int size = (pieces == 0 ? 0 : 1);
                for (; pieces > 0x3; ++size, pieces >>= 2);
                std::cout << "  " << " XO"[pieces] << " 123"[size];
                if (col < 2)
                {
                    std::cout << "  |";
                }
            }
            std::cout << std::endl;
            for (int col = 0; col < 3; ++col)
            {
                std::cout << "     " << 3 * row + col;
                if (col < 2)
                {
                    std::cout << "|";
                }
            }
            std::cout << std::endl;
            if (row < 2)
            {
                std::cout << "------|------|------" << std::endl;
            }
        }
    }

    // Play game, allowing rewind and showing optimal moves.
    void play(Arena scratch)
    {
        std::vector<State> states(1, 0);
        int turn = 1;
        while (true)
        {
            State s = states.back();
            show(turn == 1 ? s : swap(s));
            State* s_ptr = lookup(canonical(s));
            int value = unpack_value(*s_ptr);
            std::size_t moves = unpack_moves(*s_ptr);
            if (moves == 0)
            {
                if (value == 0)
                {
                    std::cout << "Game ends in a draw." << std::endl;
                }
                else
                {
                    std::cout << "Player " << (value == 1 ? turn : 3 - turn) <<
                        " wins." << std::endl;
                }
                break;
            }
            Move m{};
            while (m.start == 0 && m.end == 0)
            {
                std::cout << "Player " << turn << ", enter move " <<
                    "(-size | start, end), or (0, 0) for best move, or " <<
                    "(-1, -1) to undo move: ";
                std::cin >> m.start >> m.end;
                if (m.start == 0 && m.end == 0)
                {
                    if (value == 0)
                    {
                        std::cout << "Draw with";
                    }
                    else
                    {
                        std::cout << (value == 1 ? "Win" : "Lose") << " in " <<
                            moves << " moves with";
                    }
                    Move best = best_move(s, scratch);
                    std::cout << " (" <<
                        best.start << ", " << best.end << ")." << std::endl;
                }
            }
            if (m.start == -1 && m.end == -1)
            {
                states.pop_back();
            }
            else
            {
                states.push_back(swap(move(s, m)));
            }
            turn = 3 - turn;
        }
    }
};

int main()
{
    int num_sizes = 3;
    int num_per_size = 2;
    bool allow_move = true;
    Arena perm{1LL<<33};
    Arena scratch{1LL<<35};
    Game game{num_sizes, num_per_size, allow_move, &perm, scratch};
}

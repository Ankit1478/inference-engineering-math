# Step 20: a simple PAGED KV cache (the PagedAttention idea)
# Memory is split into small pages. Users get a page only when they need one,
# pages can be anywhere, and a page table remembers where each user's pages are.
# Run: pip install torch && python3 03_paged_kv_cache.py

import torch

torch.manual_seed(0)
d = 8              # each word = 8 numbers
PAGE_SIZE = 4      # words per page
NUM_PAGES = 6      # total pages in "GPU memory"


class PagedKVCache:
    def __init__(self):
        self.K = torch.zeros(NUM_PAGES, PAGE_SIZE, d)   # all of GPU memory, in pages
        self.V = torch.zeros(NUM_PAGES, PAGE_SIZE, d)
        self.free_pages = list(range(NUM_PAGES))        # pages nobody is using
        self.page_table = {}                            # user -> list of their pages
        self.length = {}                                # user -> words stored

    def add_word(self, user, k, v):
        pages = self.page_table.setdefault(user, [])
        n = self.length.get(user, 0)
        if n % PAGE_SIZE == 0:                          # last page full (or no page yet)
            if not self.free_pages:
                raise MemoryError("No free pages: evict or offload someone")
            pages.append(self.free_pages.pop(0))        # take a new page, from anywhere
        page, slot = pages[n // PAGE_SIZE], n % PAGE_SIZE
        self.K[page, slot] = k
        self.V[page, slot] = v
        self.length[user] = n + 1

    def get(self, user):
        pages, n = self.page_table[user], self.length[user]
        K = self.K[pages].reshape(-1, d)[:n]            # follow the page table
        V = self.V[pages].reshape(-1, d)[:n]
        return K, V

    def finish(self, user):
        self.free_pages += self.page_table.pop(user)    # give the pages back
        del self.length[user]

    def show(self):
        owner = {p: u for u, ps in self.page_table.items() for p in ps}
        print("  Memory:", " ".join(f"[{owner.get(p, '--'):>4}]" for p in range(NUM_PAGES)))
        for u, ps in self.page_table.items():
            print(f"  {u:>4}'s page table: {ps}  ({self.length[u]} words)")


cache = PagedKVCache()

print("1. Riya and Aman take turns writing words:")
for word in range(6):
    cache.add_word("Riya", torch.randn(d), torch.randn(d))
    if word < 3:
        cache.add_word("Aman", torch.randn(d), torch.randn(d))
cache.show()

print("\n2. Riya finishes, her pages are freed:")
cache.finish("Riya")
cache.show()

print("\n3. Raj arrives and writes 9 words (3 pages):")
for _ in range(9):
    cache.add_word("Raj", torch.randn(d), torch.randn(d))
cache.show()

print("\n4. Neha arrives, writes 5 words and gets Riya's old pages. Same K,V as a normal cache?")
k_list = [torch.randn(d) for _ in range(5)]
v_list = [torch.randn(d) for _ in range(5)]
for k, v in zip(k_list, v_list):
    cache.add_word("Neha", k, v)
K, V = cache.get("Neha")
print("  Neha's pages:", cache.page_table["Neha"], "(not next to each other)")
print("  Same?", torch.equal(K, torch.stack(k_list)) and torch.equal(V, torch.stack(v_list)))

def gen():
    for i in range(5):
        try:
            if i == 2:
                raise ValueError("error at 2")
            yield i
        except ValueError as e:
            print(f"内部捕获: {e}")

g = gen()
while True:
    try:
        print("得到:", next(g))
    # except ValueError as e:
    #     print(f"外部捕获: {e}")
        # 可以继续 next，但生成器仍然活着
    except StopIteration:
        print("迭代结束")
        break

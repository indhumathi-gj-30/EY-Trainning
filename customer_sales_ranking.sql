SELECT
    cust.customer_id,
    CONCAT(cust.first_name, ' ', cust.last_name) AS full_name,
    COUNT(DISTINCT ord.order_id) AS order_count,
    SUM(item.quantity * item.unit_price) AS total_revenue,
    RANK() OVER (ORDER BY SUM(item.quantity * item.unit_price) DESC) AS revenue_rank
FROM shopey.customers cust
JOIN shopey.orders ord
    ON cust.customer_id = ord.customer_id
JOIN shopey.order_lines item
    ON ord.order_id = item.order_id
GROUP BY cust.customer_id, cust.first_name, cust.last_name
HAVING COUNT(DISTINCT ord.order_id) >= 1
ORDER BY revenue_rank ASC;